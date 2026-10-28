(function () {
  var CONTACT_EMAIL = 'saidharshan922@gmail.com';

  document.documentElement.classList.add('sd-js');

  document.addEventListener('DOMContentLoaded', function () {
    // Scroll fade-in
    var els = [].slice.call(document.querySelectorAll('.sd-reveal'));
    function show(el) { el.classList.add('is-in'); }
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) { show(e.target); io.unobserve(e.target); } });
      }, { threshold: 0.12 });
      els.forEach(function (el) { io.observe(el); });
      setTimeout(function () { els.forEach(show); }, 4000);
    } else { els.forEach(show); }

    // Mobile menu
    var btn = document.getElementById('menu-btn');
    var nav = document.getElementById('site-nav');
    function setOpen(open) {
      nav.classList.toggle('is-open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    btn.addEventListener('click', function () { setOpen(!nav.classList.contains('is-open')); });
    nav.addEventListener('click', function (ev) { if (ev.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && nav.classList.contains('is-open')) { setOpen(false); btn.focus(); }
    });

    // Contact form: opens the visitor's email app with the message filled in (home page only)
    var form = document.getElementById('contact-form');
    if (form) form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var n = form.elements.name.value.trim(), e = form.elements.email.value.trim(), m = form.elements.message.value.trim();
      var body = m + '\n\n— ' + n + (e ? ' (' + e + ')' : '');
      window.location.href = 'mailto:' + CONTACT_EMAIL + '?subject=' + encodeURIComponent('Hello from ' + (n || 'your portfolio')) + '&body=' + encodeURIComponent(body);
      document.getElementById('sent').textContent = 'Opening your email app with the message filled in.';
    });

    // Footer year
    var y = document.getElementById('year');
    if (y) y.textContent = new Date().getFullYear();
  });
})();
