# Hypersphere HVAE (vMF) Interactive Visualizer

Interactive 3D sphere (Three.js) to explore how vMF concentration (kappa) and different loss weights (HVAE, Geometric, Classify) affect point distributions on the unit sphere.

## Features
- 3D unit sphere with orbit controls
- 4 classes with distinct colors; points lie exactly on the sphere surface
- Adjustable kappa (vMF), learning-rate, noise
- Adjustable loss weights: HVAE, Geometric, Classify
- Draggable class mean directions (arrow helpers)
- Metrics: avg intra-class angle, avg inter-class angle between means, cosine-based silhouette-like score
- Dataset markers: ▲ HPC, ■ Power, ● Traffic
- Start/Pause, Step, Re-initialize, Reset view
- All browser-only (no backend)

## How to run
Open `hypersphere/index.html` in your browser. If the browser blocks local file textures, start a simple HTTP server.

### Option 1: VS Code Live Server extension

### Option 2: Python http.server
```bash
# from the project root or hypersphere folder
python -m http.server 8000
# then open http://localhost:8000/hypersphere/index.html
``