/**
 * Main JavaScript for Web Summarizer app
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    
    // Add timestamp formatter
    if (typeof Intl !== 'undefined') {
        // Format timestamps in a user-friendly way
        const timestamps = document.querySelectorAll('.timestamp');
        timestamps.forEach(el => {
            const timestamp = parseInt(el.getAttribute('data-timestamp'));
            if (!isNaN(timestamp)) {
                const date = new Date(timestamp * 1000);
                el.textContent = new Intl.DateTimeFormat(navigator.language, {
                    dateStyle: 'medium',
                    timeStyle: 'short'
                }).format(date);
            }
        });
    }
    
    // Function to handle API search (for potential AJAX searching)
    window.apiSearch = async function(query, depth = 3) {
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    depth: depth
                })
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Search error:', error);
            return { error: error.message };
        }
    };
    
    // Handle form submission if we want to make it AJAX in the future
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        // We're keeping the normal form submission for now
        // but this is where we would add AJAX functionality
        searchForm.addEventListener('submit', function(e) {
            // Add loading state to button
            const button = document.getElementById('search-button');
            if (button) {
                button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
                button.disabled = true;
            }
            
            // Continue with normal form submission
            return true;
        });
    }
    
    // Add filter functionality on results page if needed
    const filterInput = document.getElementById('filter-sources');
    if (filterInput) {
        filterInput.addEventListener('input', function() {
            const filterValue = this.value.toLowerCase();
            const sourceItems = document.querySelectorAll('.list-group-item-action');
            
            sourceItems.forEach(item => {
                const title = item.querySelector('h5').textContent.toLowerCase();
                const snippet = item.querySelector('.source-snippet').textContent.toLowerCase();
                
                if (title.includes(filterValue) || snippet.includes(filterValue)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
});

// Add custom filter for date formatting
// This would be handled on the backend in a real app
// For now, we add a simple helper function
function formatDate(timestamp) {
    if (!timestamp) return '';
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}