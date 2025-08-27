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

// Class means (unit directions) - start separated for immediate visualization
const means = [
  new THREE.Vector3(1, 1, 1).normalize(),     // Positive octant
  new THREE.Vector3(-1, -1, -1).normalize(),  // Opposite corner
  new THREE.Vector3(1, -1, 1).normalize(),    // Another corner  
  new THREE.Vector3(-1, 1, -1).normalize(),   // Fourth corner
];

// Points: each has position, classId, datasetId
let points = [];

function initPoints(){
  points = [];
  
  // Mobile optimization: reduce point count on slower devices
  const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  const actualPerClass = isMobile ? Math.min(state.perClass, 150) : state.perClass; // Limit points on mobile
  
  // Create initial mixed clustering - all classes start close together
  // This simulates real data where different classes might initially be mixed
  const mixedCenters = [
    new THREE.Vector3(0.8, 0.6, 0.1).normalize(),  // Mixed region 1
    new THREE.Vector3(-0.3, 0.9, 0.2).normalize(), // Mixed region 2
    new THREE.Vector3(0.1, -0.7, 0.7).normalize(), // Mixed region 3
  ];
  
  for (let c = 0; c < state.classes; c++){
    for (let i = 0; i < actualPerClass; i++){
      const ds = i % 3; // 0:HPC, 1:Power, 2:Traffic
      
      // Initially sample around mixed centers instead of class means
      // This creates the effect where different classes start close together
      const mixedCenter = mixedCenters[i % mixedCenters.length];
      const p = sampleVMF(mixedCenter, state.kappa * 2); // Higher concentration for mixed regions
      
      points.push({pos: p, cls: c, ds});
    }
  }
  console.log(`✅ Generated ${points.length} points with initial mixed clustering (mobile optimized: ${isMobile})`);
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
  // Since means are already separated from the beginning, we don't need auto-separation
  // This function is kept for compatibility but returns false to skip separation
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
      
      // Much slower and gentler attraction toward class mean
      const classMean = means[pt.cls];
      const attractionForce = classMean.clone().sub(pt.pos).multiplyScalar(0.02); // Much slower attraction
      
      // Reduced noise for smoother movement
      const noise = randomUnitVec().multiplyScalar(state.noise * 0.2); // Much less noise
      
      // Gentle combined forces with smooth transitions
      const newPos = pt.pos.clone()
        .addScaledVector(g, -0.008)  // Much reduced gradient influence
        .add(attractionForce)        // Slow gentle attraction
        .add(noise);                 // Minimal noise
        
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
  
  // Update iteration count
  const iterationEl = document.getElementById('iterationCount');
  if (iterationEl) {
    iterationEl.textContent = state.timeStep;
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
    
    // Mobile optimization: reduce antialias and pixel ratio for better performance
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const pixelRatio = isMobile ? Math.min(window.devicePixelRatio, 1.5) : Math.min(window.devicePixelRatio, 2);
    
    renderer = new THREE.WebGLRenderer({
      canvas, 
      antialias: !isMobile, // Disable antialias on mobile for better performance
      alpha: true,
      powerPreference: isMobile ? "low-power" : "high-performance"
    });
    
    renderer.setPixelRatio(pixelRatio);
    
    // Set consistent light theme background
    renderer.setClearColor(0xffffff, 1.0);
    
    onResize();
    window.addEventListener('resize', onResize);
    console.log('✅ Renderer created with mobile optimizations');
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
    
    // Mobile-optimized controls
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enablePan = true;
    
    // Mobile-specific touch settings
    if (isMobile) {
      controls.touches = {
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN
      };
      controls.enableKeys = false; // Disable keyboard on mobile
      controls.zoomSpeed = 0.6; // Slower zoom for better control
      controls.rotateSpeed = 0.8; // Slightly slower rotation
      controls.panSpeed = 0.8; // Slower pan for precision
    }

    // Enhanced lighting system for better data point visibility
    const amb = new THREE.AmbientLight('#f8fafc', 0.9); // Brighter ambient
    scene.add(amb);
    
    const dir = new THREE.DirectionalLight('#ffffff', 0.6); // Softer directional
    dir.position.set(3, 3, 5);
    scene.add(dir);
    
    // Additional fill light for better sphere visibility
    const fillLight = new THREE.DirectionalLight('#f1f5f9', 0.4); // Brighter fill
    fillLight.position.set(-2, -2, -3);
    scene.add(fillLight);
    
    // Back light for rim effect
    const backLight = new THREE.DirectionalLight('#e2e8f0', 0.3);
    backLight.position.set(0, 0, -5);
    scene.add(backLight);

    // Mobile-optimized sphere geometry
    const sphereDetail = isMobile ? 128 : 256; // Reduced detail on mobile
    const geom = new THREE.SphereGeometry(1, sphereDetail, sphereDetail);
    
    // Main sphere with improved visibility for data points
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0xE8F4FD, // Lighter blue for better contrast
      metalness: 0.02,
      roughness: 0.3, // More roughness for better point visibility
      transparent: true,
      opacity: 0.08, // Much more transparent
      side: THREE.DoubleSide,
      envMapIntensity: 0.1,
      transmission: 0.05,
      thickness: 0.2,
      clearcoat: 0.1,
      clearcoatRoughness: 0.5,
      ior: 1.2,
      reflectivity: 0.1
    });
    sphereMesh = new THREE.Mesh(geom, mat);
    scene.add(sphereMesh);
    
    // Enhanced wireframe with better visibility
    const wireframeDetail = isMobile ? 32 : 64; // Reduced wireframe detail on mobile
    const wireframeGeom = new THREE.SphereGeometry(1.001, wireframeDetail, wireframeDetail);
    const wireframeMat = new THREE.MeshBasicMaterial({
      color: 0x4A90E2, // Lighter blue for better contrast
      wireframe: true,
      transparent: true,
      opacity: 0.25, // Slightly more visible
      linewidth: 1.2
    });
    const wireframeMesh = new THREE.Mesh(wireframeGeom, wireframeMat);
    scene.add(wireframeMesh);
    
    // Subtle rim lighting
    const rimDetail = isMobile ? 48 : 96; // Reduced rim detail on mobile
    const rimGeom = new THREE.SphereGeometry(1.004, rimDetail, rimDetail);
    const rimMat = new THREE.MeshBasicMaterial({
      color: 0x6B9DDE, // Softer blue
      transparent: true,
      opacity: 0.05, // Very subtle
      side: THREE.BackSide
    });
    const rimMesh = new THREE.Mesh(rimGeom, rimMat);
    scene.add(rimMesh);

    pointGroup = new THREE.Group();
    scene.add(pointGroup);

    // Mobile-optimized mean arrows (larger and more visible)
    for (let i=0;i<4;i++){
      const col = classColors[i];
      const dir = means[i].clone();
      const arrowLength = isMobile ? 1.4 : 1.2; // Longer arrows on mobile
      const arrowHeadLength = isMobile ? 0.3 : 0.2; // Larger arrow heads on mobile
      const arrowHeadWidth = isMobile ? 0.2 : 0.1; // Wider arrow heads on mobile
      
      const arrow = new THREE.ArrowHelper(
        dir.clone(), 
        new THREE.Vector3(0,0,0), 
        arrowLength, 
        col.getHex(),
        arrowHeadLength,
        arrowHeadWidth
      );
      
      // Make arrow lines thicker on mobile
      if (isMobile && arrow.line && arrow.line.material) {
        arrow.line.material.linewidth = 3;
      }
      
      meanArrows.push(arrow);
      scene.add(arrow);
    }
    
    console.log('✅ Scene created with mobile optimizations and', meanArrows.length, 'arrows');
  } catch (error) {
    console.error('❌ Error creating scene:', error);
    throw error;
  }
}

function spriteFor(ds, color){
  const size = 56; // Larger size for better visibility
  const cvs = document.createElement('canvas');
  cvs.width = cvs.height = size;
  const ctx = cvs.getContext('2d');
  
  // Anti-aliasing for smoother sprites
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  
  ctx.clearRect(0,0,size,size);
  
  // Enhanced outline for better visibility
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 2.5;
  ctx.fillStyle = color.getStyle();
  
  const r = 18; // Larger radius
  ctx.translate(size/2, size/2);
  
  // Add subtle shadow for depth
  ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
  ctx.shadowBlur = 3;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  if (ds===0){
    // triangle - more pronounced
    ctx.beginPath();
    ctx.moveTo(-r, r*0.8);
    ctx.lineTo(0, -r);
    ctx.lineTo(r, r*0.8);
    ctx.closePath();
    ctx.fill(); 
    ctx.shadowColor = 'transparent'; // Remove shadow for stroke
    ctx.stroke();
  } else if (ds===1){
    // square - slightly rounded corners
    ctx.beginPath();
    ctx.roundRect(-r, -r, 2*r, 2*r, 2);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.stroke();
  } else {
    // circle - perfect circle
    ctx.beginPath();
    ctx.arc(0,0,r,0,Math.PI*2);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.stroke();
  }
  
  const tex = new THREE.CanvasTexture(cvs);
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    alphaTest: 0.1,
    sizeAttenuation: true
  });
  
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.08, 0.08, 0.08); // Slightly larger for better visibility
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

// Touch and mouse interaction handling - mobile optimized
let dragging = false, dragIndex = -1;
let lastTouchTime = 0;

function onPointerDown(e){
  // Handle both mouse and touch events
  e.preventDefault();
  
  const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
  const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
  
  const {x, y, hitIdx} = pickArrow(clientX, clientY);
  if (hitIdx >= 0){ 
    dragging = true; 
    dragIndex = hitIdx; 
    controls.enabled = false;
    
    // Visual feedback - make arrow brighter when dragging
    if (meanArrows[hitIdx]) {
      const arrow = meanArrows[hitIdx];
      arrow.setColor(0xffffff); // Make it white/bright when dragging
    }
    
    // Add haptic feedback on mobile
    if (navigator.vibrate && e.touches) {
      navigator.vibrate(50); // Short vibration feedback
    }
    
    // Prevent touch scrolling
    if (e.touches) {
      document.body.style.overflow = 'hidden';
    }
    
    console.log(`🎯 Started dragging arrow ${hitIdx}`);
  }
}

function onPointerUp(e){ 
  e.preventDefault();
  
  // Restore original arrow color
  if (dragging && dragIndex >= 0 && meanArrows[dragIndex]) {
    const arrow = meanArrows[dragIndex];
    arrow.setColor(classColors[dragIndex].getHex()); // Restore original color
  }
  
  dragging = false; 
  dragIndex = -1; 
  controls.enabled = true;
  
  // Re-enable scrolling
  document.body.style.overflow = '';
  
  console.log('🎯 Stopped dragging');
}

function onPointerMove(e){
  if (!dragging) return;
  e.preventDefault();
  
  const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
  const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
  
  const p = ndcToSphere(clientX, clientY);
  if (p){ 
    means[dragIndex] = p; 
    updateMeanArrows(); 
  }
}

// Touch-specific handlers
function onTouchStart(e){
  const currentTime = Date.now();
  const timeDiff = currentTime - lastTouchTime;
  
  // Handle double tap
  if (timeDiff < 300 && timeDiff > 0) {
    // Double tap detected - could add special functionality here
    console.log('Double tap detected');
  }
  
  lastTouchTime = currentTime;
  onPointerDown(e);
}

function onTouchEnd(e){
  onPointerUp(e);
}

function onTouchMove(e){
  onPointerMove(e);
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
  // Enhanced picking for touch devices with much larger hit areas
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((clientY - rect.top) / rect.height) * 2 + 1;
  const proj = new THREE.Vector3();
  let best = -1, bestD = 1e9;
  
  // Much larger hit area for mobile devices
  const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  const hitThreshold = isMobile ? 0.15 : 0.08; // Much larger touch targets on mobile
  
  for (let i = 0; i < means.length; i++){
    proj.copy(means[i]).project(camera);
    const dx = proj.x - x, dy = proj.y - y;
    const d = dx*dx + dy*dy;
    if (d < bestD){ 
      bestD = d; 
      best = i; 
    }
  }
  
  if (bestD < hitThreshold) {
    console.log(`🎯 Arrow ${best} picked on ${isMobile ? 'mobile' : 'desktop'} (distance: ${Math.sqrt(bestD).toFixed(3)})`);
    return {x, y, hitIdx: best};
  }
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
      console.log('🔄 Points reinitialized - means already separated from start');
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
      console.log('▶️ Animation started - means already separated from beginning');
    });

    btnPause.addEventListener('click', ()=>{
      state.running = false;
      updateButtons();
      console.log('⏸️ Animation paused');
    });

    updateButtons();

    // Enhanced event binding for mobile and desktop
    if (renderer && renderer.domElement) {
      const canvas = renderer.domElement;
      
      // Mouse events (desktop)
      canvas.addEventListener('mousedown', onPointerDown, {passive: false});
      window.addEventListener('mouseup', onPointerUp, {passive: false});
      window.addEventListener('mousemove', onPointerMove, {passive: false});
      
      // Touch events (mobile)
      canvas.addEventListener('touchstart', onTouchStart, {passive: false});
      canvas.addEventListener('touchend', onTouchEnd, {passive: false});
      canvas.addEventListener('touchmove', onTouchMove, {passive: false});
      
      // Prevent context menu on long press
      canvas.addEventListener('contextmenu', (e) => e.preventDefault());
      
      // Prevent zoom on double tap
      canvas.addEventListener('touchstart', (e) => {
        if (e.touches.length > 1) {
          e.preventDefault();
        }
      }, {passive: false});
      
      console.log('✅ Enhanced mobile and desktop events bound');
    }
    
    console.log('✅ UI initialized');
  } catch (error) {
    console.error('❌ Error setting up UI:', error);
    throw error;
  }
}

function syncMeshes(){
  for (const pt of points){
    if (pt.mesh) {
      // Smooth interpolation for position updates
      const currentPos = pt.mesh.position;
      const targetPos = pt.pos;
      
      // Linear interpolation for smooth movement (lerp)
      const smoothFactor = 0.15; // Smooth transition factor
      currentPos.lerp(targetPos, smoothFactor);
      
      // Ensure mesh stays on sphere surface
      currentPos.normalize();
    }
  }
}

// ======= Main Loop =======
let lastFrameTime = 0;
let frameCount = 0;

function animate(currentTime = 0){
  try {
    requestAnimationFrame(animate);
    
    // Mobile optimization: throttle frame rate on slower devices
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const targetFPS = isMobile ? 30 : 60; // Lower FPS on mobile for better performance
    const frameInterval = 1000 / targetFPS;
    
    if (currentTime - lastFrameTime < frameInterval) {
      return; // Skip this frame
    }
    
    lastFrameTime = currentTime;
    frameCount++;
    
    if (controls) {
      controls.update();
    }
    
    if (state.running && points && points.length > 0) { 
      // Reduce computation frequency for smoother movement
      const computationFrequency = isMobile ? 0.3 : 0.6; // Much slower updates
      
      if (Math.random() < computationFrequency) {
        stepPoints(1); 
        
        // Increment time step counter less frequently for smoother visualization
        if (Math.random() < 0.5) { // Only increment every other computation
          state.timeStep++;
        }
        
        // Update means much less frequently to create chasing effect
        // Points update every frame, means update very rarely
        if (Math.random() < 0.01) { // Very infrequent mean updates
          updateMeans();
        }
      }
      
      // Always sync meshes for smooth animation
      syncMeshes(); 
      
      // Update metrics less frequently on mobile
      const metricsUpdateFrequency = isMobile ? 0.02 : 0.05; // Less frequent updates
      if (Math.random() < metricsUpdateFrequency) {
        updateMetrics(); 
      }
    }
    
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
    
    // Performance monitoring
    if (frameCount % 300 === 0) { // Log every 300 frames
      console.log(`📊 Performance: Running at ~${targetFPS}fps target`);
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
