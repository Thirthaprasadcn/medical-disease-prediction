/* Cinematic WebGL stage for MIR — Three.js */
(function () {
  const canvas = document.getElementById("webgl");
  if (!canvas || typeof THREE === "undefined") return;

  const page = document.body.dataset.page || "home";
  const isReport = page === "output" || document.body.classList.contains("is-report");
  const isHome = page === "home";
  const isUpload = page === "upload";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0xf0f4f7, 0.028);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0.35, isHome ? 5.0 : isReport ? 6.4 : 5.5);

  const root = new THREE.Group();
  scene.add(root);

  // Soft studio lights — bright clinical look
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));
  const key = new THREE.DirectionalLight(0xffffff, 1.05);
  key.position.set(3, 5, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xb8d4ce, 0.45);
  fill.position.set(-4, 1, -2);
  scene.add(fill);
  const rim = new THREE.PointLight(0x2bb7a0, 1.1, 20);
  rim.position.set(0, 0.2, 2.5);
  scene.add(rim);

  // Floor disc (soft ground)
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(3.6, 64),
    new THREE.MeshStandardMaterial({
      color: 0xf3f6f8,
      metalness: 0.04,
      roughness: 0.9,
      transparent: true,
      opacity: 0.72,
    })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -1.35;
  root.add(floor);

  // MRI / CT gantry ring
  const gantry = new THREE.Mesh(
    new THREE.TorusGeometry(1.55, 0.08, 24, 100),
    new THREE.MeshStandardMaterial({
      color: 0x2a3d48,
      metalness: 0.72,
      roughness: 0.32,
    })
  );
  gantry.rotation.x = Math.PI / 2.15;
  root.add(gantry);

  const gantryInner = new THREE.Mesh(
    new THREE.TorusGeometry(1.35, 0.025, 16, 100),
    new THREE.MeshStandardMaterial({
      color: 0x0f8f78,
      metalness: 0.6,
      roughness: 0.2,
      emissive: 0x0f8f78,
      emissiveIntensity: 0.35,
    })
  );
  gantryInner.rotation.x = Math.PI / 2.15;
  root.add(gantryInner);

  // Floating tomographic slices
  const sliceGroup = new THREE.Group();
  root.add(sliceGroup);

  function makeSliceTexture(seed) {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 256;
    const ctx = c.getContext("2d");
    const g = ctx.createRadialGradient(128, 128, 20, 128, 128, 120);
    g.addColorStop(0, "#c8d4dc");
    g.addColorStop(0.45, "#6b7c88");
    g.addColorStop(0.72, "#2a3842");
    g.addColorStop(1, "#0b1216");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(128, 128, 118, 0, Math.PI * 2);
    ctx.fill();

    // organic denser regions
    for (let i = 0; i < 18; i++) {
      const x = 80 + ((seed * 37 + i * 47) % 96);
      const y = 70 + ((seed * 53 + i * 29) % 110);
      const r = 8 + (i % 5) * 3;
      const rg = ctx.createRadialGradient(x, y, 0, x, y, r);
      rg.addColorStop(0, "rgba(230,240,245,0.55)");
      rg.addColorStop(1, "rgba(230,240,245,0)");
      ctx.fillStyle = rg;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // finding markers
    ctx.strokeStyle = "rgba(15,143,120,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(150 + (seed % 20), 120 + (seed % 15), 14, 0, Math.PI * 2);
    ctx.stroke();

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  const slices = [];
  const sliceCount = isHome ? 8 : isReport ? 5 : 7;
  for (let i = 0; i < sliceCount; i++) {
    const mat = new THREE.MeshStandardMaterial({
      map: makeSliceTexture(i + 3),
      transparent: true,
      opacity: 0.42,
      metalness: 0.1,
      roughness: 0.55,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(new THREE.CircleGeometry(1.05, 48), mat);
    mesh.position.y = -0.75 + i * 0.22;
    mesh.rotation.x = -0.12;
    sliceGroup.add(mesh);
    slices.push(mesh);
  }

  // Scan beam plane
  const beam = new THREE.Mesh(
    new THREE.PlaneGeometry(2.4, 0.03),
    new THREE.MeshBasicMaterial({
      color: 0x2bb7a0,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    })
  );
  beam.position.z = 0.02;
  root.add(beam);

  // Voxel particle cloud
  const pCount = isHome ? 1100 : isReport ? 380 : 800;
  const positions = new Float32Array(pCount * 3);
  for (let i = 0; i < pCount; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = 1.1 + Math.random() * 1.8;
    positions[i * 3] = Math.cos(a) * r;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 2.4;
    positions[i * 3 + 2] = Math.sin(a) * r;
  }
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const points = new THREE.Points(
    pGeo,
    new THREE.PointsMaterial({
      color: 0x0c7c69,
      size: 0.016,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
    })
  );
  root.add(points);

  // Outer wire sphere
  const wire = new THREE.Mesh(
    new THREE.IcosahedronGeometry(2.35, 1),
    new THREE.MeshBasicMaterial({
      color: 0x5a7280,
      wireframe: true,
      transparent: true,
      opacity: 0.1,
    })
  );
  root.add(wire);

  let scanning = false;
  let intensity = 1;
  let pulse = 0;
  const mouse = { x: 0, y: 0 };
  const targetRot = { x: 0, y: 0 };

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener("resize", resize);

  window.addEventListener(
    "pointermove",
    (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = (e.clientY / window.innerHeight) * 2 - 1;
    },
    { passive: true }
  );

  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    const speed = reduceMotion ? 0.15 : 1;

    targetRot.y = mouse.x * 0.35;
    targetRot.x = mouse.y * 0.18;
    root.rotation.y += (targetRot.y - root.rotation.y) * 0.04;
    root.rotation.x += (targetRot.x - root.rotation.x) * 0.04;

    const spin = isHome ? 1.35 : isUpload ? 1.0 : 0.7;
    gantry.rotation.z = t * 0.25 * speed * spin;
    gantryInner.rotation.z = -t * 0.45 * speed * spin;
    wire.rotation.y = t * 0.08 * speed * spin;
    wire.rotation.x = t * 0.05 * speed;
    points.rotation.y = t * 0.12 * speed * spin;
    if (isHome) root.rotation.y += 0.0015 * speed;

    slices.forEach((s, i) => {
      s.position.y = -0.75 + i * 0.22 + Math.sin(t * 1.2 * speed + i * 0.55) * 0.04 * intensity;
      s.material.opacity = 0.28 + 0.2 * Math.sin(t * 1.5 + i) * (scanning ? 1.2 : 1);
      s.rotation.z = Math.sin(t * 0.3 + i) * 0.04;
    });

    const beamY = -0.9 + ((t * (scanning ? 1.8 : 0.7) * speed) % 1.8);
    beam.position.y = beamY;
    beam.material.opacity = scanning ? 0.85 : 0.4;

    if (pulse > 0) {
      pulse *= 0.92;
      rim.intensity = 1.4 + pulse * 3;
      gantryInner.material.emissiveIntensity = 0.35 + pulse;
    } else {
      rim.intensity = 1.2 + Math.sin(t * 2) * 0.2;
      gantryInner.material.emissiveIntensity = 0.3 + Math.sin(t * 3) * 0.1;
    }

    camera.position.x += (mouse.x * 0.35 - camera.position.x) * 0.03;
    camera.position.y += (0.35 - mouse.y * 0.25 - camera.position.y) * 0.03;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
  }
  animate();

  window.MIR3D = {
    setScanning(on) {
      scanning = !!on;
      if (on) pulse = 1;
    },
    pulse() {
      pulse = 1;
    },
    setIntensity(v) {
      intensity = Math.max(0.4, Math.min(2, v));
    },
  };
})();
