#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>
#include <GL/glut.h>

using namespace std;


struct Vec2 {
    float x, y;
    Vec2(float x = 0, float y = 0) : x(x), y(y) {}

    Vec2 operator+(const Vec2& v) const {
        return Vec2(x + v.x, y + v.y);
    }
    Vec2 operator-(const Vec2& v) const {
        return Vec2(x - v.x, y - v.y);
    }
    Vec2 operator*(float s) const {
        return Vec2(x * s, y * s);
    }
    float length() const {
        return sqrt(x * x + y * y);
    }
    Vec2 normalized() const {
        float l = length();
        return l > 0 ? Vec2(x/l, y/l) : Vec2();
    }
};

struct Particle {
    Vec2 pos, vel, force;
    float mass;
    int gridX, gridY;
};

struct SimParams {
    float gravity = 500.0f;
    float attraction = 1.0f;
    float cohesion = 15.0f;
    float viscosity = 20.0f;
    float damping = 0.98f;
    float rest_density = 800.0f;
    float gas_constant = 150.0f;
    float smoothing_radius = 18.0f;
    float particle_radius = 3.0f;
    float base_time_step = 0.016f; // cuz 2^4 and syncs also smae rate
    float time_scale = 1.0f;
    float collision_force = 20000.0f;
    float collision_distance = 16.0f;
    float separation_force = 50000.0f;
    float separation_distance = 7.0f;
    float min_distance = 6.0f;
    float position_correction_strength = 0.8f;
    float mouse_force = 25000.0f;
    float mouse_radius = 60.0f;
    
    float get_time_step() const {
        return base_time_step * time_scale;
    }
};


vector<Particle> particles;
vector<vector<int>> grid;
SimParams params;
int grid_width, grid_height, grid_size;
float world_width = 800.0f, world_height = 600.0f;

Vec2 mouse_pos(0, 0);
bool mouse_pressed = false;
int mouse_button = -1;
bool spawn_at_mouse = false;
int spawn_count = 10;
bool spacebar_held = false;
mt19937 rng(69);
uniform_real_distribution<float> rand_float(0.0f, 1.0f);


void init_grid() {
    grid_size = (int)params.smoothing_radius;
    grid_width = (int)(world_width / grid_size) + 1;
    grid_height = (int)(world_height / grid_size) + 1;
    grid.resize(grid_width * grid_height);
}

void clear_grid() {
    for (auto& cell : grid) {
        cell.clear();
    }
}

void update_grid() {
    clear_grid();
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        p.gridX = max(0, min(grid_width - 1, (int)(p.pos.x / grid_size)));
        p.gridY = max(0, min(grid_height - 1, (int)(p.pos.y / grid_size)));
        grid[p.gridY * grid_width + p.gridX].push_back((int)i);
    }
}

float smoothing_kernel(float r, float h) {
    if (r >= h) return 0.0f;
    float q = 1.0f - r / h;
    return 315.0f / (64.0f * M_PI * pow(h, 9)) * pow(q, 3);
}

float smoothing_kernel_derivative(float r, float h) {
    if (r >= h || r == 0) return 0.0f;
    float q = 1.0f - r / h;
    return -945.0f / (32.0f * M_PI * pow(h, 9)) * pow(q, 2);
}

void compute_density_pressure() {
    vector<float> densities(particles.size());
    
    #pragma omp parallel for schedule(dynamic, 64) // dont ask me why 64
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        float density = 0.0f;
        
        int gx = p.gridX, gy = p.gridY;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = gx + dx, ny = gy + dy;
                if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) continue;
                
                for (int j : grid[ny * grid_width + nx]) {
                    Vec2 diff = p.pos - particles[j].pos;
                    float dist = diff.length();
                    if (dist < params.smoothing_radius) {
                        density += particles[j].mass * smoothing_kernel(dist, params.smoothing_radius);
                    }
                }
            }
        }
        densities[i] = max(density, params.rest_density * 0.01f);
    }
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        p.force = Vec2(0, -params.gravity * p.mass);
        
        float density_ratio = densities[i] / params.rest_density;
        float pressure_i = params.gas_constant * (densities[i] - params.rest_density);
        if (density_ratio > 1.0f) {
            pressure_i += params.gas_constant * pow(density_ratio - 1.0f, 2.0f) * 500.0f;
        }
        
        int gx = p.gridX, gy = p.gridY;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = gx + dx, ny = gy + dy;
                if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) continue;
                
                for (int j : grid[ny * grid_width + nx]) {
                    if (i == (size_t)j) continue;
                    
                    Vec2 diff = p.pos - particles[j].pos;
                    float dist = diff.length();
                    if (dist < params.smoothing_radius && dist > 0.01f) {
                        Vec2 dir = diff.normalized();
                        float density_ratio_j = densities[j] / params.rest_density;
                        float pressure_j = params.gas_constant * (densities[j] - params.rest_density);
                        if (density_ratio_j > 1.0f) {
                            pressure_j += params.gas_constant * pow(density_ratio_j - 1.0f, 2.0f) * 500.0f;
                        }
                        
                        float pressure_force = -(pressure_i + pressure_j) * 0.5f * particles[j].mass * smoothing_kernel_derivative(dist, params.smoothing_radius) / densities[j];
                        p.force = p.force + dir * pressure_force;
                        
                        Vec2 vel_diff = particles[j].vel - p.vel;
                        float viscosity_force = params.viscosity * particles[j].mass * smoothing_kernel_derivative(dist, params.smoothing_radius) / densities[j];
                        p.force = p.force + vel_diff * viscosity_force;
                        
                        if (dist < params.smoothing_radius * 0.6f) {
                            float cohesion_force = params.cohesion * particles[j].mass / densities[j];
                            p.force = p.force - dir * cohesion_force;
                        }
                        
                        float attraction_force = params.attraction * particles[j].mass / (dist * dist + 0.1f);
                        p.force = p.force - dir * attraction_force;
                        
                        if (dist < params.separation_distance) {
                            float overlap = params.separation_distance - dist;
                            float separation_magnitude = params.separation_force * pow(overlap / params.separation_distance, 3.0f);
                            Vec2 separation_force_vec = dir * separation_magnitude;
                            p.force = p.force + separation_force_vec;
                            
                            if (dist < params.min_distance) {
                                float extra_force = params.separation_force * 2.0f;
                                p.force = p.force + dir * extra_force;
                            }
                            
                            Vec2 relative_vel = p.vel - particles[j].vel;
                            float damping_factor = 1.2f;
                            Vec2 damping_force = relative_vel * (-damping_factor * particles[j].mass);
                            p.force = p.force + damping_force;
                        }
                        else if (dist < params.collision_distance) {
                            float overlap = params.collision_distance - dist;
                            float collision_magnitude = params.collision_force * overlap / params.collision_distance;
                            Vec2 collision_force_vec = dir * collision_magnitude;
                            p.force = p.force + collision_force_vec;
                            
                            Vec2 relative_vel = p.vel - particles[j].vel;
                            float damping_factor = 0.6f;
                            Vec2 damping_force = relative_vel * (-damping_factor * particles[j].mass);
                            p.force = p.force + damping_force;
                        }
                    }
                }
            }
        }
    }
}

void apply_mouse_forces() {
    if (!mouse_pressed) {
        return;
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        Vec2 diff = p.pos - mouse_pos;
        float dist = diff.length();
        
        if (dist < params.mouse_radius && dist > 0.1f) {
            Vec2 dir = diff.normalized();
            float force_magnitude = params.mouse_force * (1.0f - dist / params.mouse_radius);
            
            if (mouse_button == GLUT_LEFT_BUTTON) {
                p.force = p.force - dir * force_magnitude;
            } else if (mouse_button == GLUT_RIGHT_BUTTON) {
                p.force = p.force + dir * force_magnitude;
            }
        }
    }
}

void correct_positions() {
    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        
        int gx = p.gridX, gy = p.gridY;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = gx + dx, ny = gy + dy;
                if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) continue;
                
                for (int j : grid[ny * grid_width + nx]) {
                    if (i >= (size_t)j) continue;
                    
                    Particle& other = particles[j];
                    Vec2 diff = p.pos - other.pos;
                    float dist = diff.length();
                    
                    if (dist < params.min_distance && dist > 0.001f) {
                        // pos correction
                        Vec2 dir = diff.normalized();
                        float overlap = params.min_distance - dist;
                        Vec2 correction = dir * (overlap * 0.5f * params.position_correction_strength);
                        
                        // SEND EM APARTTTTTT
                        p.pos = p.pos + correction;
                        other.pos = other.pos - correction;
                        
                        // bro was goign jiggly too much so dmapeninggg
                        Vec2 avg_vel = (p.vel + other.vel) * 0.5f;
                        p.vel = p.vel * 0.8f + avg_vel * 0.2f;
                        other.vel = other.vel * 0.8f + avg_vel * 0.2f;
                    }
                }
            }
        }
    }
}

void integrate() {
    float dt = params.get_time_step();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        Vec2 acc = p.force * (1.0f / p.mass);
        
        p.vel = p.vel + acc * dt;
        p.vel = p.vel * params.damping;
        
        float max_vel = 200.0f * params.time_scale;
        float vel_mag = p.vel.length();
        if (vel_mag > max_vel) {
            p.vel = p.vel * (max_vel / vel_mag);
        }
        
        p.pos = p.pos + p.vel * dt;
        
        float restitution = 0.4f;
        float margin = params.particle_radius + 2.0f;
        
        if (p.pos.x < margin) {
            p.pos.x = margin;
            p.vel.x = abs(p.vel.x) * restitution;
        }
        if (p.pos.x > world_width - margin) {
            p.pos.x = world_width - margin;
            p.vel.x = -abs(p.vel.x) * restitution;
        }
        if (p.pos.y < margin) {
            p.pos.y = margin;
            p.vel.y = abs(p.vel.y) * restitution * 0.5f;
            p.vel.x *= 0.8f;
        }
        if (p.pos.y > world_height - margin) {
            p.pos.y = world_height - margin;
            p.vel.y = -abs(p.vel.y) * restitution;
        }
    }
}

void init_particles(int count) {
    particles.clear();
    particles.reserve(count);
    
    int cols = min(30, (int)sqrt(count) + 1);
    float spacing = params.particle_radius * 2.1f;
    
    for (int i = 0; i < count; ++i) {
        int row = i / cols;
        int col = i % cols;
        
        float x = col * spacing + world_width * 0.4f + rand_float(rng) * spacing * 0.3f;
        float y = world_height - row * spacing - 100.0f;
        
        Particle p;
        p.pos = Vec2(x, y);
        p.vel = Vec2((rand_float(rng) - 0.5f) * 5.0f, (rand_float(rng) - 0.5f) * 2.0f);
        p.mass = 1.0f;
        particles.push_back(p);
    }
}

void spawn_particles(Vec2 spawn_pos, int count) {
    if (particles.size() >= 15000) {
        cout << "[WARN] Maximum particle limit reached (15000)" << endl;
        return;
    }
    
    float spacing = params.particle_radius * 2.0f;
    int cols = min(5, count);
    
    for (int i = 0; i < count; ++i) {
        int row = i / cols;
        int col = i % cols;
        
        float x = spawn_pos.x + (col - cols/2.0f) * spacing + rand_float(rng) * spacing * 0.2f;
        float y = spawn_pos.y + row * spacing + rand_float(rng) * spacing * 0.2f;
        
        x = max(params.particle_radius, min(world_width - params.particle_radius, x));
        y = max(params.particle_radius, min(world_height - params.particle_radius, y));
        
        Particle p;
        p.pos = Vec2(x, y);
        p.vel = Vec2((rand_float(rng) - 0.5f) * 20.0f, (rand_float(rng) - 0.5f) * 10.0f);
        p.mass = 1.0f;
        particles.push_back(p);
    }
    
    cout << "[INFO] Spawned " << count << " particles at (" << spawn_pos.x << ", " << spawn_pos.y << "). Total: " << particles.size() << endl;
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    
    glColor3f(0.3f, 0.7f, 1.0f);
    glPointSize(params.particle_radius * 2.0f);
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        glVertex2f(p.pos.x / world_width * 2.0f - 1.0f, p.pos.y / world_height * 2.0f - 1.0f);
    }
    glEnd();
    
    glutSwapBuffers();
}

void update() {
    update_grid();
    compute_density_pressure();
    apply_mouse_forces();
    integrate();
    
    correct_positions();
    
    update_grid();
    
    static int frame_count = 0;
    frame_count++;
    
    if (spawn_at_mouse && mouse_pressed && frame_count % 20 == 0) {
        spawn_particles(mouse_pos, 2);
    }
    
    if (spacebar_held && frame_count % 15 == 0) {
        spawn_particles(mouse_pos, 69);
    }
    
    glutPostRedisplay();
}

void keyboard(unsigned char key, int, int) {
    switch (key) {
        case 'g': params.gravity += 100.0f; cout << "[INFO] Gravity: " << params.gravity << endl; break;
        case 'G': params.gravity = max(0.0f, params.gravity - 100.0f); cout << "[INFO] Gravity: " << params.gravity << endl; break;
        case 'a': params.attraction += 1.0f; cout << "[INFO] Attraction: " << params.attraction << endl; break;
        case 'A': params.attraction = max(0.0f, params.attraction - 1.0f); cout << "[INFO] Attraction: " << params.attraction << endl; break;
        case 'c': params.cohesion += 5.0f; cout << "[INFO] Cohesion: " << params.cohesion << endl; break;
        case 'C': params.cohesion = max(0.0f, params.cohesion - 5.0f); cout << "[INFO] Cohesion: " << params.cohesion << endl; break;
        case 'v': params.viscosity += 1.0f; cout << "[INFO] Viscosity: " << params.viscosity << endl; break;
        case 'V': params.viscosity = max(0.0f, params.viscosity - 1.0f); cout << "[INFO] Viscosity: " << params.viscosity << endl; break;
        case 'p': params.gas_constant += 20.0f; cout << "[INFO] Pressure: " << params.gas_constant << endl; break;
        case 'P': params.gas_constant = max(0.0f, params.gas_constant - 20.0f); cout << "[INFO] Pressure: " << params.gas_constant << endl; break;
        case 'f': params.collision_force += 500.0f; cout << "[INFO] Collision Force: " << params.collision_force << endl; break;
        case 'F': params.collision_force = max(0.0f, params.collision_force - 500.0f); cout << "[INFO] Collision Force: " << params.collision_force << endl; break;
        case 'd': params.collision_distance += 0.5f; cout << "[INFO] Collision Distance: " << params.collision_distance << endl; break;
        case 'D': params.collision_distance = max(3.0f, params.collision_distance - 0.5f); cout << "[INFO] Collision Distance: " << params.collision_distance << endl; break;
        case 'x': params.separation_force += 2000.0f; cout << "[INFO] Separation Force: " << params.separation_force << endl; break;
        case 'X': params.separation_force = max(1000.0f, params.separation_force - 2000.0f); cout << "[INFO] Separation Force: " << params.separation_force << endl; break;
        case 'z': params.separation_distance += 0.2f; cout << "[INFO] Separation Distance: " << params.separation_distance << endl; break;
        case 'Z': params.separation_distance = max(3.0f, params.separation_distance - 0.2f); cout << "[INFO] Separation Distance: " << params.separation_distance << endl; break;
        case 'b': params.min_distance += 0.2f; cout << "[INFO] Min Distance: " << params.min_distance << endl; break;
        case 'B': params.min_distance = max(4.0f, params.min_distance - 0.2f); cout << "[INFO] Min Distance: " << params.min_distance << endl; break;
        case 'y': params.position_correction_strength += 0.1f; cout << "[INFO] Position Correction: " << params.position_correction_strength << endl; break;
        case 'Y': params.position_correction_strength = max(0.1f, params.position_correction_strength - 0.1f); cout << "[INFO] Position Correction: " << params.position_correction_strength << endl; break;
        case 'm': params.mouse_force += 1000.0f; cout << "[INFO] Mouse Force: " << params.mouse_force << endl; break;
        case 'M': params.mouse_force = max(0.0f, params.mouse_force - 1000.0f); cout << "[INFO] Mouse Force: " << params.mouse_force << endl; break;
        case 'n': params.mouse_radius += 5.0f; cout << "[INFO] Mouse Radius: " << params.mouse_radius << endl; break;
        case 'N': params.mouse_radius = max(5.0f, params.mouse_radius - 5.0f); cout << "[INFO] Mouse Radius: " << params.mouse_radius << endl; break;
        case '1': params.time_scale = 1.0f; cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case '2': params.time_scale = 2.0f; cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case '3': params.time_scale = 3.0f; cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case '5': params.time_scale = 5.0f; cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case '0': params.time_scale = 0.5f; cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case '+': params.time_scale = min(10.0f, params.time_scale + 0.5f); cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case '-': params.time_scale = max(0.1f, params.time_scale - 0.5f); cout << "[INFO] Time Scale: " << params.time_scale << "x" << endl; break;
        case 's': spawn_particles(mouse_pos, spawn_count); break;
        case 'S': spawn_particles(Vec2(world_width * 0.5f, world_height - 50.0f), spawn_count); break;
        case 'q': spawn_count = min(50, spawn_count + 5); cout << "[INFO] Spawn Count: " << spawn_count << endl; break;
        case 'Q': spawn_count = max(1, spawn_count - 5); cout << "[INFO] Spawn Count: " << spawn_count << endl; break;
        case 't': spawn_at_mouse = !spawn_at_mouse; cout << "[INFO] Spawn at mouse: " << (spawn_at_mouse ? "ON" : "OFF") << endl; break;
        case ' ': 
            spacebar_held = true;
            spawn_particles(mouse_pos, 69);
            cout << "[INFO] Spacebar pressed - spawning 69 particles" << endl;
            break;
        case 'r': init_particles(particles.size()); cout << "[INFO] Reset particles" << endl; break;
        case 27: exit(0); break;
    }
}

void keyboard_up(unsigned char key, int, int) {
    switch (key) {
        case ' ':
            spacebar_held = false;
            cout << "[INFO] Spacebar released" << endl;
            break;
    }
}

Vec2 screen_to_world(int x, int y) {
    float world_x = (float)x;
    float world_y = world_height - (float)y;
    return Vec2(world_x, world_y);
}

void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        mouse_pressed = true;
        mouse_button = button;
        mouse_pos = screen_to_world(x, y);
        
        cout << "[INFO] Mouse " << (button == GLUT_LEFT_BUTTON ? "LEFT" : "RIGHT") << " pressed at (" << mouse_pos.x << ", " << mouse_pos.y << ")" << endl;
    } else if (state == GLUT_UP) {
        mouse_pressed = false;
        mouse_button = -1;
    }
}

void motion(int x, int y) {
    if (mouse_pressed) {
        mouse_pos = screen_to_world(x, y);
    }
}

void print_controls() {
    cout << "[INFO] Controls:" << endl;
    cout << "  g/G: Increase/Decrease gravity" << endl;
    cout << "  a/A: Increase/Decrease attraction" << endl;
    cout << "  c/C: Increase/Decrease cohesion" << endl;
    cout << "  v/V: Increase/Decrease viscosity" << endl;
    cout << "  p/P: Increase/Decrease pressure" << endl;
    cout << "  f/F: Increase/Decrease collision force" << endl;
    cout << "  d/D: Increase/Decrease collision distance" << endl;
    cout << "  x/X: Increase/Decrease separation force" << endl;
    cout << "  z/Z: Increase/Decrease separation distance" << endl;
    cout << "  b/B: Increase/Decrease minimum distance" << endl;
    cout << "  y/Y: Increase/Decrease position correction strength" << endl;
    cout << "  m/M: Increase/Decrease mouse force" << endl;
    cout << "  n/N: Increase/Decrease mouse radius" << endl;
    cout << "  TIME SCALING:" << endl;
    cout << "    1/2/3/5: Set time scale to 1x/2x/3x/5x" << endl;
    cout << "    0: Set time scale to 0.5x (slow motion)" << endl;
    cout << "    +/-: Increase/Decrease time scale by 0.5x" << endl;
    cout << "  PARTICLE SPAWNING:" << endl;
    cout << "    s: Spawn particles at mouse position" << endl;
    cout << "    S: Spawn particles at top center" << endl;
    cout << "    SPACEBAR: Spawn 69 particles (hold for continuous)" << endl;
    cout << "    q/Q: Increase/Decrease spawn count" << endl;
    cout << "    t: Toggle spawn at mouse mode" << endl;
    cout << "  r: Reset particles" << endl;
    cout << "  Left Click: Attract particles to mouse" << endl;
    cout << "  Right Click: Repel particles from mouse" << endl;
    cout << "  ESC: Exit" << endl;
}

int main(int argc, char** argv) {
    int particle_count = 1069;
    
    if (argc == 2) {
        particle_count = atoi(argv[1]);
        if (particle_count <= 0 || particle_count > 10000) {
            cout << "[ERROR] Particle count must be between 1 and 10000" << endl;
            return 1;
        }
    } else if (argc > 2) {
        cout << "[ERROR] Usage: " << argv[0] << " [particle_count]" << endl;
        cout << "[INFO] Default particle count is 1069 if no argument provided" << endl;
        return 1;
    }
    
    cout << "[INFO] Initializing fluid simulation with " << particle_count << " particles" << endl;
    
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    cout << "[INFO] Using OpenMP with " << num_threads << " threads" << endl;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize((int)world_width, (int)world_height);
    glutCreateWindow("FluSim");
    
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    init_grid();
    init_particles(particle_count);
    print_controls();
    
    glutDisplayFunc(display);
    glutIdleFunc(update);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboard_up);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(motion);
    
    cout << "[SUCCESS] Simulation started" << endl;
    glutMainLoop();
    
    return 0;
}
