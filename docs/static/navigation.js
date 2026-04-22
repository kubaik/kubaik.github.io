/* ============================================
   ENHANCED NAVIGATION JAVASCRIPT
   ============================================ */

document.addEventListener('DOMContentLoaded', function () {

    // ============================================
    // 1. STICKY HEADER WITH SHADOW ON SCROLL
    // ============================================
    const header = document.querySelector('header');

    if (header) {
        window.addEventListener('scroll', function () {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

            if (scrollTop > 10) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        });
    }


    // ============================================
    // 2. ACTIVE PAGE INDICATOR
    // ============================================
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');

    navLinks.forEach(link => {
        const linkPath = new URL(link.href).pathname;

        if (currentPath === linkPath ||
            (currentPath.includes(linkPath) && linkPath !== '/')) {
            link.classList.add('active');
        }
    });


    // ============================================
    // 3. MOBILE MENU FUNCTIONALITY
    // ============================================

    function createMobileMenuElements() {
        const nav = document.querySelector('nav');
        const container = document.querySelector('header .container');

        if (!nav || !container) return;

        if (!document.querySelector('.mobile-menu-toggle')) {
            const toggleButton = document.createElement('button');
            toggleButton.className = 'mobile-menu-toggle';
            toggleButton.innerHTML = '☰';
            toggleButton.setAttribute('aria-label', 'Toggle mobile menu');
            container.appendChild(toggleButton);
        }

        if (!document.querySelector('.mobile-menu-close')) {
            const closeButton = document.createElement('button');
            closeButton.className = 'mobile-menu-close';
            closeButton.innerHTML = '×';
            closeButton.setAttribute('aria-label', 'Close mobile menu');
            nav.insertBefore(closeButton, nav.firstChild);
        }

        if (!document.querySelector('.mobile-menu-overlay')) {
            const overlay = document.createElement('div');
            overlay.className = 'mobile-menu-overlay';
            document.body.appendChild(overlay);
        }
    }

    if (window.innerWidth <= 768) {
        createMobileMenuElements();
    }

    window.addEventListener('resize', function () {
        if (window.innerWidth <= 768 && !document.querySelector('.mobile-menu-toggle')) {
            createMobileMenuElements();
        }
    });

    document.addEventListener('click', function (e) {
        const nav = document.querySelector('nav');
        const overlay = document.querySelector('.mobile-menu-overlay');

        if (!nav) return;

        if (e.target.classList.contains('mobile-menu-toggle')) {
            nav.classList.add('mobile-menu-open');
            if (overlay) overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        if (e.target.classList.contains('mobile-menu-close') ||
            e.target.classList.contains('mobile-menu-overlay')) {
            nav.classList.remove('mobile-menu-open');
            if (overlay) overlay.classList.remove('active');
            document.body.style.overflow = '';
        }

        if (e.target.matches('nav a')) {
            nav.classList.remove('mobile-menu-open');
            if (overlay) overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            const nav = document.querySelector('nav');
            const overlay = document.querySelector('.mobile-menu-overlay');

            if (nav && nav.classList.contains('mobile-menu-open')) {
                nav.classList.remove('mobile-menu-open');
                if (overlay) overlay.classList.remove('active');
                document.body.style.overflow = '';
            }
        }
    });


    // ============================================
    // 4. KEYBOARD NAVIGATION ENHANCEMENT
    // ============================================
    const navElements = document.querySelectorAll('nav a, .mobile-menu-toggle');

    navElements.forEach((element, index) => {
        element.addEventListener('keydown', function (e) {
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                const nextElement = navElements[index + 1] || navElements[0];
                nextElement.focus();
            }

            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                const prevElement = navElements[index - 1] || navElements[navElements.length - 1];
                prevElement.focus();
            }
        });
    });


    // ============================================
    // 5. SMOOTH SCROLL TO ANCHOR LINKS
    // ============================================
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');

            if (href === '#') return;

            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();

                const headerHeight = header ? header.offsetHeight : 0;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - headerHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });


    // ============================================
    // 6. BACK TO TOP + READING PROGRESS BAR
    //    Works on both homepage and post pages.
    //    Button is injected once if not in the DOM.
    //    Progress bar is injected on post pages only.
    // ============================================

    // --- Reading progress bar (post pages only) ---
    const isPostPage = Boolean(
        document.querySelector('.post-content') ||
        document.querySelector('article.blog-post')
    );

    const progressBar = (function () {
        if (!isPostPage) return null;
        const bar = document.createElement('div');
        bar.id = 'reading-progress';
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('aria-valuemin', '0');
        bar.setAttribute('aria-valuemax', '100');
        bar.setAttribute('aria-valuenow', '0');
        document.body.prepend(bar);
        return bar;
    })();

    // --- Back-to-top button (injected once, works on every page) ---
    let backToTopBtn = document.getElementById('back-to-top');
    if (!backToTopBtn) {
        backToTopBtn = document.createElement('button');
        backToTopBtn.id = 'back-to-top';
        backToTopBtn.className = 'back-to-top';
        backToTopBtn.setAttribute('aria-label', 'Back to top');
        backToTopBtn.innerHTML = '<span aria-hidden="true">&#8593;</span>';
        document.body.appendChild(backToTopBtn);
    }

    function getScrollPercent() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const docHeight =
            document.documentElement.scrollHeight -
            document.documentElement.clientHeight;
        return docHeight > 0 ? Math.min(100, Math.round((scrollTop / docHeight) * 100)) : 0;
    }

    function onScrollExtras() {
        const scrollY = window.pageYOffset;

        // Show / hide back-to-top button
        if (scrollY > 300) {
            backToTopBtn.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
        }

        // Update reading progress bar (post pages only)
        if (progressBar) {
            const pct = getScrollPercent();
            progressBar.style.width = pct + '%';
            progressBar.setAttribute('aria-valuenow', pct);
        }
    }

    window.addEventListener('scroll', onScrollExtras, { passive: true });

    backToTopBtn.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Keyboard shortcut: press T (outside inputs) to scroll to top
    document.addEventListener('keydown', function (e) {
        if (e.key !== 't' && e.key !== 'T') return;
        const tag = document.activeElement && document.activeElement.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

});  // end DOMContentLoaded