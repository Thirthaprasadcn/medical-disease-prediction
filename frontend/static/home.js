/* Home page — journey pulse + 3D accent */
(function () {
  const items = document.querySelectorAll(".journey li");
  if (!items.length) return;

  let i = 0;
  setInterval(() => {
    items.forEach((el) => el.classList.remove("active"));
    items[i % items.length].classList.add("active");
    if (window.MIR3D) window.MIR3D.pulse();
    i += 1;
  }, 2200);

  if (window.MIR3D) {
    window.MIR3D.setIntensity(1.2);
    window.MIR3D.pulse();
  }
})();
