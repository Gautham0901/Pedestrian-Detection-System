/**
 * Main JavaScript file for the Pedestrian Detection System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize any global functionality here
    console.log('Pedestrian Detection System initialized');
    
    // Handle active navigation links
    const currentPath = window.location.pathname;
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
});