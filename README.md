cat > README.md << 'EOF'
# Thermo-FHN Grid: Collective Dynamics of Thermosensitive Neuronal Networks

This repository contains simulation code and analysis tools for studying the **collective dynamics of thermosensitive FitzHugh–Nagumo (FHN) neurons** under external electric field forcing and thermal noise.

The project investigates how **electric field amplitude, frequency, and temperature** control synchronization, firing variability, and spatial organization in large neuronal networks.

This work supports a research manuscript on field-controlled collective dynamics in noisy excitable media.

---

## Scientific Objective

Neuronal activity is influenced by environmental factors such as:

- External electric fields  
- Thermal fluctuations (noise)  
- Spatial coupling  

While previous studies focused on **single thermosensitive neurons**, this project studies:

- Large network (25 × 25 neurons)
- Diffusive spatial coupling
- Periodic electric field forcing
- Stochastic thermal noise
- Collective order parameters

Key analyses include:

- Kuramoto synchronization
- Phase diagrams in (Em, f)
- Temperature dependence
- Spatial firing patterns
- Mean-field dynamics

---

## Model

Each neuron follows a thermosensitive FitzHugh–Nagumo model:

dx/dt = x − x³/3 − y + Em sin(2πft) + K∇²x + √(2D)ξ(t)  
dy/dt = ε(x + a − by)

Where:

- Em : electric field amplitude  
- f : forcing frequency  
- D : noise intensity (effective temperature)  
- K : spatial coupling strength  

---

## Repository Structure
