// Chart.js configurations and data visualizations for CryptoRL Research Website

// Chart.js global configuration
Chart.defaults.font.family = 'Inter, -apple-system, BlinkMacSystemFont, sans-serif';
Chart.defaults.font.size = 12;
Chart.defaults.color = '#475569';

// Color palette
const colors = {
    primary: '#2563eb',
    secondary: '#64748b',
    accent: '#06b6d4',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    gradient: {
        primary: ['#2563eb', '#3b82f6', '#60a5fa'],
        secondary: ['#64748b', '#94a3b8', '#cbd5e1']
    }
};

// Dark mode colors
const darkColors = {
    primary: '#3b82f6',
    secondary: '#94a3b8',
    accent: '#22d3ee',
    success: '#34d399',
    warning: '#fbbf24',
    error: '#f87171',
    gradient: {
        primary: ['#3b82f6', '#60a5fa', '#93c5fd'],
        secondary: ['#94a3b8', '#cbd5e1', '#e2e8f0']
    }
};

// Utility function to detect dark mode
function isDarkMode() {
    return document.body.classList.contains('section-dark') || 
           document.documentElement.getAttribute('data-theme') === 'dark';
}

// Get appropriate colors based on theme
function getChartColors() {
    return isDarkMode() ? darkColors : colors;
}

// Cumulative Returns Chart
function createReturnsChart() {
    const ctx = document.getElementById('returnsChart');
    if (!ctx) return;

    const chartColors = getChartColors();
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024'],
            datasets: [
                {
                    label: 'Mamba-PPO + LLM',
                    data: [100, 108, 115, 125, 135, 145],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primary + '20',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'PPO Baseline',
                    data: [100, 104, 108, 113, 118, 122],
                    borderColor: chartColors.secondary,
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5
                },
                {
                    label: 'Buy & Hold',
                    data: [100, 102, 105, 108, 110, 112],
                    borderColor: chartColors.warning,
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: isDarkMode() ? '#1e293b' : '#ffffff',
                    titleColor: isDarkMode() ? '#e2e8f0' : '#1e293b',
                    bodyColor: isDarkMode() ? '#94a3b8' : '#475569',
                    borderColor: isDarkMode() ? '#334155' : '#e2e8f0',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0',
                        borderColor: isDarkMode() ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: isDarkMode() ? '#94a3b8' : '#475569'
                    }
                },
                y: {
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0',
                        borderColor: isDarkMode() ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: isDarkMode() ? '#94a3b8' : '#475569',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Portfolio Value (%)',
                        color: isDarkMode() ? '#94a3b8' : '#475569'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

// Risk-Return Scatter Plot
function createRiskReturnChart() {
    const ctx = document.getElementById('riskReturnChart');
    if (!ctx) return;

    const chartColors = getChartColors();
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Mamba-PPO + LLM',
                    data: [{x: 12.8, y: 23.4}],
                    backgroundColor: chartColors.primary,
                    borderColor: chartColors.primary,
                    pointRadius: 12,
                    pointHoverRadius: 15
                },
                {
                    label: 'Mamba-SAC + LLM',
                    data: [{x: 14.1, y: 21.7}],
                    backgroundColor: chartColors.accent,
                    borderColor: chartColors.accent,
                    pointRadius: 10,
                    pointHoverRadius: 13
                },
                {
                    label: 'PPO + LLM',
                    data: [{x: 15.3, y: 19.8}],
                    backgroundColor: chartColors.success,
                    borderColor: chartColors.success,
                    pointRadius: 8,
                    pointHoverRadius: 11
                },
                {
                    label: 'Mamba-PPO (No LLM)',
                    data: [{x: 16.9, y: 16.5}],
                    backgroundColor: chartColors.warning,
                    borderColor: chartColors.warning,
                    pointRadius: 8,
                    pointHoverRadius: 11
                },
                {
                    label: 'PPO Baseline',
                    data: [{x: 18.5, y: 15.2}],
                    backgroundColor: chartColors.secondary,
                    borderColor: chartColors.secondary,
                    pointRadius: 8,
                    pointHoverRadius: 11
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: isDarkMode() ? '#1e293b' : '#ffffff',
                    titleColor: isDarkMode() ? '#e2e8f0' : '#1e293b',
                    bodyColor: isDarkMode() ? '#94a3b8' : '#475569',
                    borderColor: isDarkMode() ? '#334155' : '#e2e8f0',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: Risk ${context.parsed.x}%, Return ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0',
                        borderColor: isDarkMode() ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: isDarkMode() ? '#94a3b8' : '#475569',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Maximum Drawdown (%)',
                        color: isDarkMode() ? '#94a3b8' : '#475569'
                    }
                },
                y: {
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0',
                        borderColor: isDarkMode() ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: isDarkMode() ? '#94a3b8' : '#475569',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Annual Return (%)',
                        color: isDarkMode() ? '#94a3b8' : '#475569'
                    }
                }
            }
        }
    });
}

// Performance Metrics Chart
function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    const chartColors = getChartColors();
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Volatility', 'Stability'],
            datasets: [
                {
                    label: 'Mamba-PPO + LLM',
                    data: [85, 90, 75, 80, 70, 85],
                    backgroundColor: chartColors.primary + '30',
                    borderColor: chartColors.primary,
                    pointBackgroundColor: chartColors.primary,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: chartColors.primary
                },
                {
                    label: 'PPO Baseline',
                    data: [60, 65, 50, 70, 60, 65],
                    backgroundColor: chartColors.secondary + '30',
                    borderColor: chartColors.secondary,
                    pointBackgroundColor: chartColors.secondary,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: chartColors.secondary
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: isDarkMode() ? '#1e293b' : '#ffffff',
                    titleColor: isDarkMode() ? '#e2e8f0' : '#1e293b',
                    bodyColor: isDarkMode() ? '#94a3b8' : '#475569',
                    borderColor: isDarkMode() ? '#334155' : '#e2e8f0',
                    borderWidth: 1
                }
            },
            scales: {
                r: {
                    angleLines: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0'
                    },
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0'
                    },
                    pointLabels: {
                        color: isDarkMode() ? '#94a3b8' : '#475569'
                    },
                    ticks: {
                        color: isDarkMode() ? '#94a3b8' : '#475569',
                        backdropColor: 'transparent'
                    }
                }
            }
        }
    });
}

// Live Data Chart (for real-time updates)
function createLiveDataChart() {
    const ctx = document.getElementById('liveDataChart');
    if (!ctx) return;

    const chartColors = getChartColors();
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Live PnL',
                data: [],
                borderColor: chartColors.primary,
                backgroundColor: chartColors.primary + '20',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    },
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0'
                    }
                },
                y: {
                    grid: {
                        color: isDarkMode() ? '#334155' : '#e2e8f0'
                    }
                }
            }
        }
    });

    return chart;
}

// Update chart data
function updateChartData(chart, newData) {
    if (!chart) return;
    
    chart.data.labels.push(new Date());
    chart.data.datasets[0].data.push(newData);
    
    // Keep only last 50 data points
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.update('none');
}

// Theme-aware chart updates
function updateChartsTheme() {
    const charts = Chart.getChart('returnsChart') || 
                   Chart.getChart('riskReturnChart') || 
                   Chart.getChart('performanceChart');
    
    if (charts) {
        charts.options.plugins.tooltip.backgroundColor = isDarkMode() ? '#1e293b' : '#ffffff';
        charts.options.plugins.tooltip.titleColor = isDarkMode() ? '#e2e8f0' : '#1e293b';
        charts.options.plugins.tooltip.bodyColor = isDarkMode() ? '#94a3b8' : '#475569';
        charts.options.plugins.tooltip.borderColor = isDarkMode() ? '#334155' : '#e2e8f0';
        
        // Update scales
        charts.options.scales.x.grid.color = isDarkMode() ? '#334155' : '#e2e8f0';
        charts.options.scales.y.grid.color = isDarkMode() ? '#334155' : '#e2e8f0';
        charts.options.scales.x.ticks.color = isDarkMode() ? '#94a3b8' : '#475569';
        charts.options.scales.y.ticks.color = isDarkMode() ? '#94a3b8' : '#475569';
        
        charts.update();
    }
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all charts
    createReturnsChart();
    createRiskReturnChart();
    createPerformanceChart();
    
    // Theme change listener
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme') {
                updateChartsTheme();
            }
        });
    });
    
    observer.observe(document.body, { attributes: true });
    observer.observe(document.documentElement, { attributes: true });
});

// Export for global use
window.CryptoRLCharts = {
    createReturnsChart,
    createRiskReturnChart,
    createPerformanceChart,
    createLiveDataChart,
    updateChartData,
    updateChartsTheme
};