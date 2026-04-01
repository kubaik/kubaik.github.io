/* ============================================
   ENHANCED NAVIGATION JAVASCRIPT
   Add this script before closing </body> tag
   ============================================ */

document.addEventListener('DOMContentLoaded', function() {
    
    // ============================================
    // 1. STICKY HEADER WITH SHADOW ON SCROLL
    // ============================================
    const header = document.querySelector('header');
    
    if (header) {
        window.addEventListener('scroll', function() {
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
        
        // Check if current page matches link
        if (currentPath === linkPath || 
            (currentPath.includes(linkPath) && linkPath !== '/')) {
            link.classList.add('active');
        }
    });
    
    
    // ============================================
    // 3. MOBILE MENU FUNCTIONALITY
    // ============================================
    
    // Create mobile menu elements if they don't exist
    function createMobileMenuElements() {
        const nav = document.querySelector('nav');
        const container = document.querySelector('header .container');
        
        if (!nav || !container) return;
        
        // Create hamburger button if it doesn't exist
        if (!document.querySelector('.mobile-menu-toggle')) {
            const toggleButton = document.createElement('button');
            toggleButton.className = 'mobile-menu-toggle';
            toggleButton.innerHTML = '☰';
            toggleButton.setAttribute('aria-label', 'Toggle mobile menu');
            container.appendChild(toggleButton);
        }
        
        // Create close button inside nav if it doesn't exist
        if (!document.querySelector('.mobile-menu-close')) {
            const closeButton = document.createElement('button');
            closeButton.className = 'mobile-menu-close';
            closeButton.innerHTML = '×';
            closeButton.setAttribute('aria-label', 'Close mobile menu');
            nav.insertBefore(closeButton, nav.firstChild);
        }
        
        // Create overlay if it doesn't exist
        if (!document.querySelector('.mobile-menu-overlay')) {
            const overlay = document.createElement('div');
            overlay.className = 'mobile-menu-overlay';
            document.body.appendChild(overlay);
        }
    }
    
    // Only create mobile menu elements on mobile devices
    if (window.innerWidth <= 768) {
        createMobileMenuElements();
    }
    
    // Recreate elements on resize if needed
    window.addEventListener('resize', function() {
        if (window.innerWidth <= 768 && !document.querySelector('.mobile-menu-toggle')) {
            createMobileMenuElements();
        }
    });
    
    // Mobile menu toggle functionality
    document.addEventListener('click', function(e) {
        const nav = document.querySelector('nav');
        const overlay = document.querySelector('.mobile-menu-overlay');
        
        if (!nav) return;
        
        // Open menu
        if (e.target.classList.contains('mobile-menu-toggle')) {
            nav.classList.add('mobile-menu-open');
            if (overlay) overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
        
        // Close menu
        if (e.target.classList.contains('mobile-menu-close') || 
            e.target.classList.contains('mobile-menu-overlay')) {
            nav.classList.remove('mobile-menu-open');
            if (overlay) overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
        
        // Close menu when clicking a nav link
        if (e.target.matches('nav a')) {
            nav.classList.remove('mobile-menu-open');
            if (overlay) overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    });
    
    // Close mobile menu on escape key
    document.addEventListener('keydown', function(e) {
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
        element.addEventListener('keydown', function(e) {
            // Navigate with arrow keys
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
        anchor.addEventListener('click', function(e) {
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
});
