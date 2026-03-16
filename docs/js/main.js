/* ============================
   TIME SERIES COUNCIL - MAIN JS
   ============================ */

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initScrollAnimations();
    initMobileNav();
    initParticles();
    initSmoothScroll();
});

/* ---- Navbar scroll effect ---- */
function initNavbar() {
    const navbar = document.getElementById('navbar');
    let ticking = false;

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                if (window.scrollY > 50) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
                ticking = false;
            });
            ticking = true;
        }
    });
}

/* ---- Scroll-triggered animations ---- */
function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('[data-animate]');

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        },
        {
            threshold: 0.1,
            rootMargin: '0px 0px -40px 0px',
        }
    );

    animatedElements.forEach((el) => observer.observe(el));
}

/* ---- Mobile navigation ---- */
function initMobileNav() {
    const toggle = document.getElementById('navToggle');
    const links = document.querySelector('.nav-links');

    if (!toggle || !links) return;

    toggle.addEventListener('click', () => {
        links.classList.toggle('active');
        toggle.classList.toggle('active');
    });

    // Close on link click
    links.querySelectorAll('a').forEach((link) => {
        link.addEventListener('click', () => {
            links.classList.remove('active');
            toggle.classList.remove('active');
        });
    });
}

/* ---- Floating particles in hero ---- */
function initParticles() {
    const container = document.getElementById('heroParticles');
    if (!container) return;

    const particleCount = 30;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        const size = Math.random() * 3 + 1;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const duration = Math.random() * 20 + 15;
        const delay = Math.random() * 10;
        const opacity = Math.random() * 0.15 + 0.03;

        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: ${Math.random() > 0.5 ? '#111111' : '#888888'};
            border-radius: 50%;
            left: ${x}%;
            top: ${y}%;
            opacity: ${opacity};
            animation: particleFloat ${duration}s ease-in-out ${delay}s infinite;
            pointer-events: none;
        `;
        container.appendChild(particle);
    }

    // Add particle animation style
    const style = document.createElement('style');
    style.textContent = `
        @keyframes particleFloat {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(${rand(-30, 30)}px, ${rand(-40, 40)}px) scale(1.2); }
            50% { transform: translate(${rand(-20, 20)}px, ${rand(-30, 30)}px) scale(0.8); }
            75% { transform: translate(${rand(-30, 30)}px, ${rand(-20, 20)}px) scale(1.1); }
        }
    `;
    document.head.appendChild(style);
}

function rand(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

/* ---- Smooth scroll for anchor links ---- */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener('click', (e) => {
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                e.preventDefault();
                const navHeight = document.getElementById('navbar').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.scrollY - navHeight - 20;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth',
                });
            }
        });
    });
}
