// HVAE + vMF interactive simulation on unit sphere
// Author: MAC Project Visualizer
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// ======= Math helpers =======
const randn = () => {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

const normalize = (v) => {
  const n = Math.hypot(v.x, v.y, v.z);
  if (n === 0) return new THREE.Vector3(0, 0, 1);
  return v.clone().divideScalar(n);
};

const randomUnitVec = () => normalize(new THREE.Vector3(randn(), randn(), randn()));

const slerp = (a, b, t) => {
  // Spherical linear interpolation on unit sphere
  const dot = Math.min(1, Math.max(-1, a.clone().dot(b)));
  const theta = Math.acos(dot) * t;
  const rel = b.clone().sub(a.clone().multiplyScalar(dot)).normalize();
  return a.clone().multiplyScalar(Math.cos(theta)).add(rel.multiplyScalar(Math.sin(theta)));
};

// Sample from von Mises–Fisher on S^2 using Wood's algorithm approximation
// Direction mu (unit vec), concentration kappa >= 0
function sampleVMF(mu, kappa){
  if (kappa <= 1e-6) {
    // near-uniform
    return randomUnitVec();
  }
  // Wood (1994) algorithm adaptation for 3D
  const b = (-2*kappa + Math.sqrt(4*kappa*kappa + 4)) / 2; // b = (-2k + sqrt(4k^2 + 4))/2
  const x0 = (1 - b) / (1 + b);
  const c = kappa * x0 + 2 * Math.log(1 - x0*x0);

  let x, u, w;
  while(true){
    const z = Math.random();
    const u1 = Math.random();
    const u2 = Math.random();
    const w1 = (1 - (1 + b) * z) / (1 - (1 - b) * z);
    const uCheck = kappa * w1 + 2 * Math.log(1 - x0*w1) - c;
    if (uCheck >= Math.log(u1)){
      x = w1; u = u2; break;
    }
  }
  const v = new THREE.Vector3(randn(), randn(), 0).normalize();
  const w_perp = v.multiplyScalar(Math.sqrt(1 - x*x));
  const w_par = new THREE.Vector3(0,0,1).multiplyScalar(x);
  const w_cart = w_perp.add(w_par);

  // rotate to mu
  const zAxis = new THREE.Vector3(0,0,1);
  const axis = zAxis.clone().cross(mu);
  const angle = Math.acos(Math.min(1, Math.max(-1, zAxis.dot(mu))));
  const q = new THREE.Quaternion().setFromAxisAngle(axis.normalize(), angle);
  return w_cart.applyQuaternion(q).normalize();
}

// Project to unit sphere (safety)
const projectToSphere = (p) => p.normalize();

// ======= Simulation state =======
const state = {
  classes: 4,
  perClass: 100,
  kappa: 15, // Higher concentration for better clustering
  hvaeW: 0.6,   // HVAE loss weight (fixed)
  geoW: 0.2,    // Geometric loss weight (fixed)
  clsW: 0.2,    // Classification loss weight (fixed)
  r: 0.8,       // Higher margin size parameter for better separation
  Dmax: 2.0,    // Maximum distance between any two points on unit sphere
  noise: 0.06,  // Noise amount for realistic dynamics
  meanLr: 0.02, // Learning rate for mean optimization
  running: false,
  timeStep: 0,  // Track time steps for automatic mean separation
  separationTriggered: false, // Flag to track if separation has started
};

const classColors = [
  new THREE.Color('#e53e3e'),
  new THREE.Color('#3182ce'),
  new THREE.Color('#38a169'),
  new THREE.Color('#d69e2e'),
];

// dataset shapes for fun (▲, ■, ● simulated via sprite shapes)
const datasetMarkers = ['triangle', 'square', 'circle'];

// Class means (unit directions) - start closer together for better visualization
const means = [
  new THREE.Vector3(0.5, 0.5, 0.707).normalize(),  // Start in similar region
  new THREE.Vector3(0.6, 0.4, 0.693).normalize(),  // Close to first
  new THREE.Vector3(0.4, 0.6, 0.693).normalize(),  // Close to others
  new THREE.Vector3(0.7, 0.3, 0.641).normalize(),  // Slightly different
];

// Points: each has position, classId, datasetId
let points = [];

function initPoints(){
  points = [];
  
  // Create initial mixed clustering - all classes start close together
  // This simulates real data where different classes might initially be mixed
  const mixedCenters = [
    new THREE.Vector3(0.8, 0.6, 0.1).normalize(),  // Mixed region 1
    new THREE.Vector3(-0.3, 0.9, 0.2).normalize(), // Mixed region 2
    new THREE.Vector3(0.1, -0.7, 0.7).normalize(), // Mixed region 3
  ];
  
  for (let c = 0; c < state.classes; c++){
    for (let i = 0; i < state.perClass; i++){
      const ds = i % 3; // 0:HPC, 1:Power, 2:Traffic
      
      // Initially sample around mixed centers instead of class means
      // This creates the effect where different classes start close together
      const mixedCenter = mixedCenters[i % mixedCenters.length];
      const p = sampleVMF(mixedCenter, state.kappa * 2); // Higher concentration for mixed regions
      
      points.push({pos: p, cls: c, ds});
    }
  }
  console.log(`✅ Generated ${points.length} points with initial mixed clustering`);
}

// Function to re-cluster existing points around their current class means
function reclusterPoints() {
  if (!points || points.length === 0) return;
  
  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    // Re-sample point position around its class mean
    const newPos = sampleVMF(means[point.cls], state.kappa);
    point.pos = newPos;
  }
  console.log(`🔄 Re-clustered ${points.length} points around updated means`);
}

// ======= Losses (based on MAC ablation study results) =======
function hvaeLoss(p, mu){
  // MOST CRITICAL: Removing HVAE drops Acc from 0.960 to 0.925 (-3.5%)
  // Enhanced concentration effect - stronger pull toward class mean
  const cos = p.dot(mu);
  const angle = Math.acos(Math.min(1, Math.max(-1, cos)));
  // Stronger exponential penalty for deviation from class mean
  return Math.pow(angle / Math.PI, 3) * (state.kappa / 3 + 2); // Increased penalty
}

function geometricLoss(p, others){
  // L_geo: geometry-aware regularization from equation (4.15)
  // L_geo = Σ[|d(zi, zj) - r · Dmax| · I(yi ≠ yj)]
  let loss = 0;
  const margin = state.r * state.Dmax; // r * Dmax where Dmax = 2
  
  for (const m of others){
    // Calculate Euclidean distance on unit sphere
    const distance = p.clone().sub(m).length();
    
    // For inter-class pairs (yi ≠ yj): enforce separation of r*Dmax
    const violation = Math.max(0, margin - distance);
    loss += violation * violation; // Quadratic penalty for violations
  }
  
  return loss / Math.max(1, others.length);
}

function classifyLoss(p, cls){
  // LEAST CRITICAL: Removing Cls drops Acc from 0.960 to 0.785 (-18.2%)
  // BUT this is offset by improved mAF1 (0.950 vs 0.913)
  // Weak classification margin - fine-tuning effect
  const pos = p;
  const toOwn = Math.acos(Math.min(1, Math.max(-1, pos.dot(means[cls]))));
  let bestOther = Math.PI; // Start with maximum angle
  for (let i=0;i<means.length;i++) if (i!==cls){
    const angleToOther = Math.acos(Math.min(1, Math.max(-1, pos.dot(means[i]))));
    bestOther = Math.min(bestOther, angleToOther);
  }
  const margin = bestOther - toOwn; // want positive, large
  // Gentle hinge loss - less aggressive than HVAE
  return Math.max(0, 0.3 - margin) * 0.3; // Much weaker than HVAE
}

// ======= Mean Optimization Functions =======
function computeMeanGradients() {
  const gradients = means.map(() => new THREE.Vector3(0, 0, 0));
  
  // Compute gradients for each class mean
  for (let c = 0; c < state.classes; c++) {
    const classPts = points.filter(p => p.cls === c);
    
    // HVAE gradient: attract points to their class mean
    for (const pt of classPts) {
      const diff = means[c].clone().sub(pt.pos);
      const dist = diff.length();
      if (dist > 1e-6) {
        // Gradient points towards better concentration
        gradients[c].add(diff.normalize().multiplyScalar(state.hvaeW * 1.5));
      }
    }
    
    // Geometric gradient: enforce separation based on parameter r
    for (let j = 0; j < means.length; j++) {
      if (j === c) continue;
      
      const diff = means[c].clone().sub(means[j]);
      const currentDist = diff.length();
      const targetDist = state.r * state.Dmax;
      
      if (currentDist < targetDist && currentDist > 1e-6) {
        // Push means apart if too close
        const pushForce = (targetDist - currentDist) / targetDist;
        gradients[c].add(diff.normalize().multiplyScalar(state.geoW * pushForce * 2.0));
      }
    }
    
    // Classification gradient: improve class separation
    for (const pt of classPts) {
      // Move our mean closer to its points
      const toPt = pt.pos.clone().sub(means[c]);
      if (toPt.length() > 1e-6) {
        gradients[c].add(toPt.normalize().multiplyScalar(state.clsW * 0.8));
      }
    }
    
    // Project gradient onto tangent space of unit sphere
    if (gradients[c].length() > 1e-6) {
      const meanNormal = means[c].clone().normalize();
      const proj = gradients[c].dot(meanNormal);
      gradients[c].sub(meanNormal.multiplyScalar(proj));
    }
  }
  
  return gradients;
}

// Function to automatically separate means to opposite positions after time threshold
function autoSeparateMeans() {
  if (state.timeStep >= 20 && !state.separationTriggered) {
    console.log('🎯 Triggering automatic mean separation after 20 time steps');
    state.separationTriggered = true;
    
    // Define target positions for 4 classes - spread them to opposite corners of sphere
    const targetPositions = [
      new THREE.Vector3(1, 1, 1).normalize(),     // Positive octant
      new THREE.Vector3(-1, -1, -1).normalize(),  // Opposite corner
      new THREE.Vector3(1, -1, 1).normalize(),    // Another corner
      new THREE.Vector3(-1, 1, -1).normalize(),   // Fourth corner
    ];
    
    // Gradually move means toward target positions
    for (let c = 0; c < state.classes; c++) {
      if (c < targetPositions.length) {
        const target = targetPositions[c];
        const current = means[c];
        
        // Use spherical interpolation for smooth movement
        const t = 0.05; // Slow interpolation factor
        means[c] = slerp(current, target, t);
        means[c].normalize();
      }
    }
    
    updateMeanArrows();
    return true; // Indicates separation is happening
  }
  return false;
}

function updateMeans() {
  // Check for automatic separation first
  const separating = autoSeparateMeans();
  
  // If separation is triggered, continue with gentle separation movement
  if (state.separationTriggered) {
    // Continue gradual separation even after initial trigger
    const targetPositions = [
      new THREE.Vector3(1, 1, 1).normalize(),     
      new THREE.Vector3(-1, -1, -1).normalize(),  
      new THREE.Vector3(1, -1, 1).normalize(),    
      new THREE.Vector3(-1, 1, -1).normalize(),   
    ];
    
    for (let c = 0; c < state.classes; c++) {
      if (c < targetPositions.length) {
        const target = targetPositions[c];
        const current = means[c];
        const distance = current.distanceTo(target);
        
        // Continue moving toward target if not close enough
        if (distance > 0.05) {
          const t = 0.02; // Very slow continued movement
          means[c] = slerp(current, target, t);
          means[c].normalize();
        }
      }
    }
    
    updateMeanArrows();
    return;
  }
  
  // Normal gradient-based mean update for first 20 steps
  const gradients = computeMeanGradients();
  let significantChange = false;
  
  // Slow down mean updates significantly for better visualization
  const slowMeanLr = state.meanLr * 0.1; // 10x slower than point updates
  
  for (let c = 0; c < state.classes; c++) {
    const oldMean = means[c].clone();
    
    // Apply much slower gradient update for means
    means[c].add(gradients[c].multiplyScalar(slowMeanLr));
    
    // Ensure mean stays on unit sphere
    means[c].normalize();
    
    // Check if change is significant (threshold for regeneration)
    const changeAngle = Math.acos(Math.min(1, Math.max(-1, oldMean.dot(means[c]))));
    if (changeAngle > 0.02) { // Lower threshold since means move slower
      significantChange = true;
    }
  }
  
  // Don't regenerate points - let them gradually move toward updated means
  // This creates the visualization effect where points chase slowly-moving means
  
  // Update arrow visualizations
  updateMeanArrows();
  
  if (significantChange) {
    console.log('🎯 Means updated slowly - points will gradually follow');
  }
}

function totalLossForPoint(pt){
  const cls = pt.cls;
  const mu = means[cls];
  const others = means.filter((_, i)=>i!==cls);
  const Lh = hvaeLoss(pt.pos, mu) * state.hvaeW;
  const Lg = geometricLoss(pt.pos, others) * state.geoW;
  const Lc = classifyLoss(pt.pos, cls) * state.clsW;
  return {Lh, Lg, Lc, L: Lh+Lg+Lc};
}

// ======= Optim step on sphere =======
function gradApprox(pt){
  // Numerical gradient on sphere using small eps, then reproject
  const eps = 1e-3;
  const base = totalLossForPoint(pt).L;
  const v = pt.pos.clone();
  const axes = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
  const g = new THREE.Vector3();
  for (const ax of axes){
    const v2 = projectToSphere(v.clone().addScaledVector(ax, eps));
    const tmp = {pos: v2, cls: pt.cls};
    const L2 = totalLossForPoint(tmp).L;
    const d = (L2 - base) / eps;
    g.addScaledVector(ax, d);
  }
  // Project gradient onto tangent plane of sphere at v
  const radial = v.clone().multiplyScalar(v.dot(g));
  const tangential = g.clone().sub(radial);
  return tangential;
}

function stepPoints(steps=1){
  for (let s=0;s<steps;s++){
    for (const pt of points){
      const g = gradApprox(pt);
      
      // Enhanced attraction toward class mean
      const classMean = means[pt.cls];
      const attractionForce = classMean.clone().sub(pt.pos).multiplyScalar(0.08); // Strong attraction
      
      // Reduced noise for more directed movement
      const noise = randomUnitVec().multiplyScalar(state.noise * 0.5);
      
      // Combined forces: gradient descent + attraction to class mean + noise
      const newPos = pt.pos.clone()
        .addScaledVector(g, -0.02)  // Reduced gradient influence
        .add(attractionForce)       // Strong class mean attraction
        .add(noise);                // Reduced noise
        
      pt.pos = projectToSphere(newPos);
    }
  }
}

// ======= Metrics =======
function degrees(rad){ return rad * 180 / Math.PI; }

function updateMetrics(){
  const C = state.classes;
  
  // Calculate d_intra: average inter-class distance (between class centroids)
  let dintra = 0;
  let intraPairs = 0;
  for (let i = 0; i < C; i++) {
    for (let j = i + 1; j < C; j++) {
      // Use Euclidean distance between means on unit sphere
      const dist = means[i].clone().sub(means[j]).length();
      dintra += dist;
      intraPairs++;
    }
  }
  dintra = intraPairs > 0 ? dintra / intraPairs : 0;
  
  // Calculate d_inter: average intra-class distance (within each class)
  let dinter = 0;
  let interPairs = 0;
  for (let c = 0; c < C; c++) {
    const classPoints = points.filter(p => p.cls === c);
    const Nc = classPoints.length;
    
    if (Nc > 1) {
      for (let k1 = 0; k1 < Nc; k1++) {
        for (let k2 = k1 + 1; k2 < Nc; k2++) {
          const dist = classPoints[k1].pos.clone().sub(classPoints[k2].pos).length();
          dinter += dist;
          interPairs++;
        }
      }
    }
  }
  dinter = interPairs > 0 ? dinter / interPairs : 0;
  
  // Calculate DQ: Data Quality ratio
  const DQ = dintra > 0 ? dinter / dintra : 0;
  
  // Calculate mean separation metrics
  let minMeanDist = Infinity;
  let maxMeanDist = 0;
  let avgMeanDist = 0;
  let meanPairCount = 0;
  
  for (let i = 0; i < C; i++) {
    for (let j = i + 1; j < C; j++) {
      const dist = means[i].distanceTo(means[j]);
      minMeanDist = Math.min(minMeanDist, dist);
      maxMeanDist = Math.max(maxMeanDist, dist);
      avgMeanDist += dist;
      meanPairCount++;
    }
  }
  avgMeanDist = meanPairCount > 0 ? avgMeanDist / meanPairCount : 0;
  
  const targetSeparation = state.r * state.Dmax;

  // Display metrics with proper formatting and r parameter effect
  const el = document.getElementById('metrics');
  if (el) {
    el.innerHTML = `
      <div><strong>Data Quality Metrics:</strong></div>
      <div>d_intra (inter-class): ${dintra.toFixed(4)}</div>
      <div>d_inter (intra-class): ${dinter.toFixed(4)}</div>
      <div>DQ ratio: ${DQ.toFixed(4)}</div>
    `;
  }
}

// ======= Three.js Scene =======
let renderer, scene, camera, controls;
let sphereMesh, pointGroup, meanArrows=[];

function makeRenderer(){
  try {
    const canvas = document.getElementById('scene');
    if (!canvas) {
      throw new Error('Canvas element "scene" not found');
    }
    
    renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Set consistent light theme background
    renderer.setClearColor(0xffffff, 1.0);
    
    onResize();
    window.addEventListener('resize', onResize);
    console.log('✅ Renderer created with light theme background');
  } catch (error) {
    console.error('❌ Error creating renderer:', error);
    throw error;
  }
}

function onResize(){
  const canvas = document.getElementById('scene');
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(200, rect.width);
  const h = Math.max(200, rect.height);
  if (!renderer) return;
  renderer.setSize(w, h, false);
  if (!camera) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function makeScene(){
  try {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(3.5, 4.5, 3.8); // Raised camera position higher
    
    if (!renderer) {
      throw new Error('Renderer not initialized');
    }
    
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enablePan = true;

    // Enhanced lighting system for consistent light theme
    const amb = new THREE.AmbientLight('#f0f4f8', 0.8);
    scene.add(amb);
    
    const dir = new THREE.DirectionalLight('#ffffff', 0.8);
    dir.position.set(3, 3, 5);
    scene.add(dir);
    
    // Additional fill light for better sphere visibility
    const fillLight = new THREE.DirectionalLight('#e2e8f0', 0.3);
    fillLight.position.set(-2, -2, -3);
    scene.add(fillLight);

    // Enhanced unit sphere with premium materials for better roundness
    const geom = new THREE.SphereGeometry(1, 256, 256); // Ultra-high resolution for smoothness
    
    // Main sphere with premium glass-like material
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x87CEEB, // Sky blue
      metalness: 0.05,
      roughness: 0.1,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
      envMapIntensity: 0.3,
      transmission: 0.1,
      thickness: 0.5,
      clearcoat: 0.3,
      clearcoatRoughness: 0.2,
      ior: 1.4,
      reflectivity: 0.2
    });
    sphereMesh = new THREE.Mesh(geom, mat);
    scene.add(sphereMesh);
    
    // Enhanced wireframe with university blue theme
    const wireframeGeom = new THREE.SphereGeometry(1.001, 64, 64);
    const wireframeMat = new THREE.MeshBasicMaterial({
      color: 0x2E4BC1, // University blue
      wireframe: true,
      transparent: true,
      opacity: 0.4,
      linewidth: 1.5
    });
    const wireframeMesh = new THREE.Mesh(wireframeGeom, wireframeMat);
    scene.add(wireframeMesh);
    
    // Premium rim lighting with university colors
    const rimGeom = new THREE.SphereGeometry(1.004, 96, 96);
    const rimMat = new THREE.MeshBasicMaterial({
      color: 0x1E3A8A,
      transparent: true,
      opacity: 0.08,
      side: THREE.BackSide
    });
    const rimMesh = new THREE.Mesh(rimGeom, rimMat);
    scene.add(rimMesh);

    pointGroup = new THREE.Group();
    scene.add(pointGroup);

    // mean arrows (draggable along sphere)
    for (let i=0;i<4;i++){
      const col = classColors[i];
      const dir = means[i].clone();
      const arrow = new THREE.ArrowHelper(dir.clone(), new THREE.Vector3(0,0,0), 1.2, col.getHex());
      meanArrows.push(arrow);
      scene.add(arrow);
    }
    
    console.log('✅ Scene created with', meanArrows.length, 'arrows');
  } catch (error) {
    console.error('❌ Error creating scene:', error);
    throw error;
  }
}

function spriteFor(ds, color){
  const size = 44; // px
  const cvs = document.createElement('canvas');
  cvs.width = cvs.height = size;
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,size,size);
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  ctx.fillStyle = color.getStyle();
  const r = 16;
  ctx.translate(size/2, size/2);
  if (ds===0){
    // triangle
    ctx.beginPath();
    ctx.moveTo(-r, r*0.8);
    ctx.lineTo(0, -r);
    ctx.lineTo(r, r*0.8);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
  } else if (ds===1){
    // square
    ctx.beginPath();
    ctx.rect(-r, -r, 2*r, 2*r);
    ctx.fill(); ctx.stroke();
  } else {
    // circle
    ctx.beginPath();
    ctx.arc(0,0,r,0,Math.PI*2);
    ctx.fill(); ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(cvs);
  const mat = new THREE.SpriteMaterial({map: tex});
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.06, 0.06, 0.06);
  return spr;
}

function rebuildPointMeshes(){
  try {
    // remove old
    while(pointGroup.children.length) pointGroup.remove(pointGroup.children[0]);
    
    if (!points || points.length === 0) {
      console.warn('⚠️ No points to rebuild');
      return;
    }
    
    // add new
    for (const pt of points){
      const color = classColors[pt.cls];
      const spr = spriteFor(pt.ds, color);
      spr.position.copy(pt.pos);
      pointGroup.add(spr);
      pt.mesh = spr;
    }
    
    console.log('✅ Rebuilt', points.length, 'point meshes');
  } catch (error) {
    console.error('❌ Error rebuilding point meshes:', error);
  }
}

function updateMeanArrows(){
  for (let i=0;i<meanArrows.length;i++){
    meanArrows[i].setDirection(means[i].clone().normalize());
  }
}

// Simple drag along sphere for mean arrows
let dragging = false, dragIndex = -1;
function onMouseDown(e){
  const {x,y, hitIdx} = pickArrow(e.clientX, e.clientY);
  if (hitIdx>=0){ dragging = true; dragIndex=hitIdx; controls.enabled = false; }
}
function onMouseUp(){ dragging=false; dragIndex=-1; controls.enabled = true; }
function onMouseMove(e){
  if (!dragging) return;
  const p = ndcToSphere(e.clientX, e.clientY);
  if (p){ means[dragIndex] = p; updateMeanArrows(); }
}

function ndcToSphere(clientX, clientY){
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((clientY - rect.top) / rect.height) * 2 + 1;
  const ray = new THREE.Raycaster();
  ray.setFromCamera({x,y}, camera);
  const hit = ray.intersectObject(sphereMesh);
  if (hit && hit[0]){
    return hit[0].point.clone().normalize();
  }
  return null;
}

function pickArrow(clientX, clientY){
  // naive pick by projecting directions; fast and sufficient
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((clientY - rect.top) / rect.height) * 2 + 1;
  const proj = new THREE.Vector3();
  let best=-1, bestD=1e9;
  for (let i=0;i<means.length;i++){
    proj.copy(means[i]).project(camera);
    const dx = proj.x - x, dy = proj.y - y;
    const d = dx*dx + dy*dy;
    if (d < bestD){ bestD=d; best=i; }
  }
  if (bestD < 0.05) return {x, y, hitIdx: best};
  return {x, y, hitIdx: -1};
}

// ======= UI Wiring =======
// ======= Report Link Management =======
function initReportLink() {
  const reportLink = document.getElementById('report-link');
  
  if (reportLink) {
    // You can update this URL to point to your actual internship report
    reportLink.href = 'https://example.com/your-internship-report.pdf'; // Replace with actual URL
    
    reportLink.addEventListener('click', (e) => {
      // Optional: Track analytics or show confirmation
      console.log('� Opening internship report');
      
      // If you want to show a confirmation before opening
      // e.preventDefault();
      // if (confirm('Open internship report in new tab?')) {
      //   window.open(reportLink.href, '_blank');
      // }
    });
    
    console.log('🔗 Report link initialized');
  } else {
    console.warn('⚠️ Report link element not found');
  }
}

// ======= Global functions for UI =======
// Auto-normalize weights to sum to 1 (fixed weights)
function normalizeWeights() {
  const total = state.hvaeW + state.geoW + state.clsW;
  if (total > 0) {
    state.hvaeW /= total;
    state.geoW /= total;
    state.clsW /= total;
  }
}

// Make function globally accessible (though not used anymore)
window.setTrainingPhase = () => {};

function setUI(){
  try {
    const bind = (id, key, fmt=(v)=>v)=>{
      const el = document.getElementById(id);
      const lab = document.getElementById(id+"Val");
      if (!el || !lab) {
        console.error(`❌ Element ${id} or ${id}Val not found`);
        return;
      }
      const sync = ()=>{ 
        lab.textContent = fmt(el.value); 
        state[key] = parseFloat(el.value);
        if (key === 'r') {
          console.log(`🔄 Parameter r updated to: ${state.r}`);
        }
      };
      el.addEventListener('input', ()=>{ 
        sync(); 
        if(key==='kappa'||key==='perClass'){ 
          initPoints(); 
          rebuildPointMeshes(); 
        } else if(key==='r') {
          // When r changes, regenerate points with updated clustering
          console.log(`🔄 Regenerating points for new r parameter: ${state.r}`);
          initPoints(); 
          rebuildPointMeshes();
          updateMeanArrows();
        }
      });
      sync();
    };
    
    bind('numPoints','perClass', v=>parseInt(v));
    bind('kappa','kappa', v=>parseInt(v));
    bind('r','r', v=>parseFloat(v)); // Margin parameter with proper parsing
    bind('meanLr','meanLr', v=>parseFloat(v)); // Mean learning rate
    bind('noise','noise');

    const btnInit = document.getElementById('btnInit');
    const btnStart = document.getElementById('btnStart');
    const btnPause = document.getElementById('btnPause');

    if (!btnInit || !btnStart || !btnPause) {
      throw new Error('One or more button elements not found');
    }

    const updateButtons = ()=>{
      btnStart.disabled = !!state.running;
      btnPause.disabled = !state.running;
      btnStart.textContent = state.running ? 'Running...' : 'Start';
    };

    btnInit.addEventListener('click', (e)=>{
      // Reset time step and separation flag
      state.timeStep = 0;
      state.separationTriggered = false;
      
      // Re-sample with current kappa; hold class means (unless Alt is pressed)
      if (e.altKey){
        for (let i=0;i<means.length;i++) means[i] = randomUnitVec();
        updateMeanArrows();
        console.log('🔄 Randomized class means');
      }
      initPoints();
      rebuildPointMeshes();
      updateMetrics();
      console.log('🔄 Points reinitialized - automatic separation will trigger after 20 steps');
    });

    btnStart.addEventListener('click', ()=>{
      if (!points || points.length===0){ 
        initPoints(); 
        rebuildPointMeshes(); 
      }
      // Reset time step when starting
      state.timeStep = 0;
      state.separationTriggered = false;
      
      state.running = true;
      updateButtons();
      console.log('▶️ Animation started - automatic separation will trigger after 20 steps');
    });

    btnPause.addEventListener('click', ()=>{
      state.running = false;
      updateButtons();
      console.log('⏸️ Animation paused');
    });

    updateButtons();

    if (renderer && renderer.domElement) {
      renderer.domElement.addEventListener('mousedown', onMouseDown);
      window.addEventListener('mouseup', onMouseUp);
      window.addEventListener('mousemove', onMouseMove);
      console.log('✅ Mouse events bound');
    }
    
    console.log('✅ UI initialized');
  } catch (error) {
    console.error('❌ Error setting up UI:', error);
    throw error;
  }
}

function syncMeshes(){
  for (const pt of points){
    if (pt.mesh) pt.mesh.position.copy(pt.pos);
  }
}

// ======= Main Loop =======
function animate(){
  try {
    requestAnimationFrame(animate);
    
    if (controls) {
      controls.update();
    }
    
    if (state.running && points && points.length > 0) { 
      stepPoints(1); 
      
      // Increment time step counter
      state.timeStep++;
      
      // Update means much less frequently to create chasing effect
      // Points update every frame, means update every ~50 frames
      if (Math.random() < 0.02) {
        updateMeans();
      }
      
      syncMeshes(); 
      
      // Update metrics every 10 frames to reduce overhead
      if (Math.random() < 0.1) {
        updateMetrics(); 
      }
    }
    
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  } catch (error) {
    console.error('❌ Error in animation loop:', error);
    // Continue the loop even if there's an error
    requestAnimationFrame(animate);
  }
}

// ======= Boot =======
function initApp() {
  try {
    makeRenderer();
    makeScene();
    initPoints();
    rebuildPointMeshes();
    updateMeanArrows();
    setUI();
    updateMetrics();
    animate();
    console.log('✅ Hypersphere app initialized successfully');
  } catch (error) {
    console.error('❌ Error initializing app:', error);
    document.body.innerHTML = `<div style="color: red; padding: 20px;">
      <h2>Application Initialization Error</h2>
      <p>Details: ${error.message}</p>
      <p>Please reload the page or check the console.</p>
    </div>`;
  }
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    try {
      initReportLink(); // Initialize report link
      initApp();
    } catch (error) {
      console.error('❌ Critical error during initialization:', error);
    }
  });
} else {
  initApp();
}
