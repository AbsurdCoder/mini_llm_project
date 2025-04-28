// Main JavaScript for Mini LLM UI

$(document).ready(function() {
    // Training data upload
    $('#training-data-form').submit(function(e) {
        e.preventDefault();
        
        var formData = new FormData(this);
        
        $.ajax({
            url: '/api/upload-training-data/',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#training-data-upload-status').html('<div class="status-success">Training data uploaded successfully!</div>');
                
                // Add new row to table
                var newRow = '<tr>' +
                    '<td>' + response.name + '</td>' +
                    '<td>' + formatFileSize(response.file_size) + '</td>' +
                    '<td>' + response.uploaded_at + '</td>' +
                    '<td><button class="btn btn-sm btn-danger delete-training-data" data-id="' + response.id + '">Delete</button></td>' +
                    '</tr>';
                
                // Remove "no data" row if it exists
                $('#no-training-data-row').remove();
                
                // Add new row
                $('#training-data-table tbody').prepend(newRow);
                
                // Reset form
                $('#training-data-form')[0].reset();
            },
            error: function(xhr) {
                var errorMsg = 'Error uploading training data.';
                if (xhr.responseJSON && xhr.responseJSON.errors) {
                    errorMsg = JSON.stringify(xhr.responseJSON.errors);
                }
                $('#training-data-upload-status').html('<div class="status-error">' + errorMsg + '</div>');
            }
        });
    });
    
    // Tokenizer upload/creation
    $('#tokenizer-form').submit(function(e) {
        e.preventDefault();
        
        var formData = new FormData(this);
        
        $.ajax({
            url: '/api/upload-tokenizer/',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#tokenizer-upload-status').html('<div class="status-success">Tokenizer created/uploaded successfully!</div>');
                
                // Add new row to table
                var newRow = '<tr>' +
                    '<td>' + response.name + '</td>' +
                    '<td>' + response.tokenizer_type + '</td>' +
                    '<td>' + response.vocab_size + '</td>' +
                    '<td>' + new Date().toLocaleString() + '</td>' +
                    '<td><button class="btn btn-sm btn-danger delete-tokenizer" data-id="' + response.id + '">Delete</button></td>' +
                    '</tr>';
                
                // Remove "no data" row if it exists
                $('#no-tokenizer-row').remove();
                
                // Add new row
                $('#tokenizer-table tbody').prepend(newRow);
                
                // Reset form
                $('#tokenizer-form')[0].reset();
            },
            error: function(xhr) {
                var errorMsg = 'Error creating/uploading tokenizer.';
                if (xhr.responseJSON && xhr.responseJSON.errors) {
                    errorMsg = JSON.stringify(xhr.responseJSON.errors);
                }
                $('#tokenizer-upload-status').html('<div class="status-error">' + errorMsg + '</div>');
            }
        });
    });
    
    // Model configuration
    $('#model-config-form').submit(function(e) {
        e.preventDefault();
        
        var formData = $(this).serialize();
        
        $.ajax({
            url: '/api/create-model/',
            type: 'POST',
            data: formData,
            success: function(response) {
                $('#model-config-status').html('<div class="status-success">Model created successfully!</div>');
                
                // Add new row to table
                var newRow = '<tr>' +
                    '<td>' + response.name + '</td>' +
                    '<td>' + response.model_type + '</td>' +
                    '<td>' + (response.tokenizer || 'None') + '</td>' +
                    '<td>-</td>' +
                    '<td><span class="badge bg-secondary">No</span></td>' +
                    '<td>' + new Date().toLocaleString() + '</td>' +
                    '<td>' +
                    '<div class="btn-group">' +
                    '<button class="btn btn-sm btn-primary train-model" data-id="' + response.id + '">Train</button>' +
                    '<button class="btn btn-sm btn-danger delete-model" data-id="' + response.id + '">Delete</button>' +
                    '</div>' +
                    '</td>' +
                    '</tr>';
                
                // Remove "no data" row if it exists
                $('#no-model-row').remove();
                
                // Add new row
                $('#model-table tbody').prepend(newRow);
                
                // Reset form
                $('#model-config-form')[0].reset();
            },
            error: function(xhr) {
                var errorMsg = 'Error creating model.';
                if (xhr.responseJSON && xhr.responseJSON.errors) {
                    errorMsg = JSON.stringify(xhr.responseJSON.errors);
                }
                $('#model-config-status').html('<div class="status-error">' + errorMsg + '</div>');
            }
        });
    });
    
    // Model upload
    $('#model-upload-form').submit(function(e) {
        e.preventDefault();
        
        var formData = new FormData(this);
        
        $.ajax({
            url: '/api/upload-model/',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#model-upload-status').html('<div class="status-success">Model uploaded successfully!</div>');
                
                // Add new row to table
                var newRow = '<tr>' +
                    '<td>' + response.name + '</td>' +
                    '<td>' + response.model_type + '</td>' +
                    '<td>' + (response.tokenizer || 'None') + '</td>' +
                    '<td>-</td>' +
                    '<td><span class="badge bg-success">Yes</span></td>' +
                    '<td>' + new Date().toLocaleString() + '</td>' +
                    '<td>' +
                    '<div class="btn-group">' +
                    '<button class="btn btn-sm btn-primary train-model" data-id="' + response.id + '">Train</button>' +
                    '<button class="btn btn-sm btn-danger delete-model" data-id="' + response.id + '">Delete</button>' +
                    '</div>' +
                    '</td>' +
                    '</tr>';
                
                // Remove "no data" row if it exists
                $('#no-model-row').remove();
                
                // Add new row
                $('#model-table tbody').prepend(newRow);
                
                // Reset form
                $('#model-upload-form')[0].reset();
            },
            error: function(xhr) {
                var errorMsg = 'Error uploading model.';
                if (xhr.responseJSON && xhr.responseJSON.errors) {
                    errorMsg = JSON.stringify(xhr.responseJSON.errors);
                }
                $('#model-upload-status').html('<div class="status-error">' + errorMsg + '</div>');
            }
        });
    });
    
    // Delete training data
    $(document).on('click', '.delete-training-data', function() {
        var dataId = $(this).data('id');
        var row = $(this).closest('tr');
        
        if (confirm('Are you sure you want to delete this training data?')) {
            $.ajax({
                url: '/api/training-data/' + dataId + '/delete/',
                type: 'POST',
                success: function() {
                    row.remove();
                    
                    // Add "no data" row if table is empty
                    if ($('#training-data-table tbody tr').length === 0) {
                        $('#training-data-table tbody').html('<tr id="no-training-data-row"><td colspan="4" class="text-center">No training data available</td></tr>');
                    }
                },
                error: function() {
                    alert('Error deleting training data.');
                }
            });
        }
    });
    
    // Delete tokenizer
    $(document).on('click', '.delete-tokenizer', function() {
        var tokenizerId = $(this).data('id');
        var row = $(this).closest('tr');
        
        if (confirm('Are you sure you want to delete this tokenizer?')) {
            $.ajax({
                url: '/api/tokenizers/' + tokenizerId + '/delete/',
                type: 'POST',
                success: function() {
                    row.remove();
                    
                    // Add "no data" row if table is empty
                    if ($('#tokenizer-table tbody tr').length === 0) {
                        $('#tokenizer-table tbody').html('<tr id="no-tokenizer-row"><td colspan="5" class="text-center">No tokenizers available</td></tr>');
                    }
                },
                error: function() {
                    alert('Error deleting tokenizer.');
                }
            });
        }
    });
    
    // Delete model
    $(document).on('click', '.delete-model', function() {
        var modelId = $(this).data('id');
        var row = $(this).closest('tr');
        
        if (confirm('Are you sure you want to delete this model?')) {
            $.ajax({
                url: '/api/models/' + modelId + '/delete/',
                type: 'POST',
                success: function() {
                    row.remove();
                    
                    // Add "no data" row if table is empty
                    if ($('#model-table tbody tr').length === 0) {
                        $('#model-table tbody').html('<tr id="no-model-row"><td colspan="7" class="text-center">No models available</td></tr>');
                    }
                },
                error: function() {
                    alert('Error deleting model.');
                }
            });
        }
    });
    
    // Alternative approach for opening training modal
    $(document).on('click', '.train-model', function() {
        var modelId = $(this).data('id');
        $('#training-model-id').val(modelId);
        
        // Try multiple approaches to show the modal
        try {
            // Approach 1: jQuery method
            $('#training-config-modal').modal('show');
        } catch (e) {
            try {
                // Approach 2: Bootstrap 5 constructor
                var trainingModal = new bootstrap.Modal(document.getElementById('training-config-modal'));
                trainingModal.show();
            } catch (e2) {
                // Approach 3: Direct attribute manipulation
                $('#training-config-modal').addClass('show').css('display', 'block');
                $('body').addClass('modal-open').append('<div class="modal-backdrop fade show"></div>');
            }
        }
    });

    // Add a close handler for the manual approach
    $(document).on('click', '[data-bs-dismiss="modal"]', function() {
        $(this).closest('.modal').removeClass('show').css('display', 'none');
        $('body').removeClass('modal-open');
        $('.modal-backdrop').remove();
    });
    
    // Start training
    $('#start-training-btn').click(function() {
        var modelId = $('#training-model-id').val();
        var formData = $('#training-session-form').serialize();
        
        $.ajax({
            url: '/api/models/' + modelId + '/train/',
            type: 'POST',
            data: formData,
            success: function(response) {
                // Close modal using multiple approaches
                try {
                    $('#training-config-modal').modal('hide');
                } catch (e) {
                    $('#training-config-modal').removeClass('show').css('display', 'none');
                    $('body').removeClass('modal-open');
                    $('.modal-backdrop').remove();
                }
                
                // Show success message
                alert('Training started successfully! Session ID: ' + response.session_id);
                
                // Open training progress modal
                viewTrainingProgress(response.session_id);
                
                // Reset form
                $('#training-session-form')[0].reset();
            },
            error: function(xhr) {
                var errorMsg = 'Error starting training.';
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMsg = xhr.responseJSON.message;
                }
                alert(errorMsg);
            }
        });
    });
    
    // View training progress
    $(document).on('click', '.view-training-progress', function() {
        var sessionId = $(this).data('id');
        viewTrainingProgress(sessionId);
    });
    
    // Text generation
    $('#text-generation-form').submit(function(e) {
        e.preventDefault();
        
        var formData = $(this).serialize();
        $('#generation-status').html('<div class="status-info">Generating text...</div>');
        
        $.ajax({
            url: '/api/generate-text/',
            type: 'POST',
            data: formData,
            success: function(response) {
                $('#generation-status').html('<div class="status-success">Text generated successfully!</div>');
                $('#generated-text-container').html('<pre>' + response.generated_text + '</pre>');
            },
            error: function(xhr) {
                var errorMsg = 'Error generating text.';
                if (xhr.responseJSON && xhr.responseJSON.errors) {
                    errorMsg = JSON.stringify(xhr.responseJSON.errors);
                }
                $('#generation-status').html('<div class="status-error">' + errorMsg + '</div>');
            }
        });
    });
    
    // Helper function to view training progress
    function viewTrainingProgress(sessionId) {
        // Show modal using multiple approaches
        try {
            // Approach 1: jQuery method
            $('#training-progress-modal').modal('show');
        } catch (e) {
            try {
                // Approach 2: Bootstrap 5 constructor
                var progressModal = new bootstrap.Modal(document.getElementById('training-progress-modal'));
                progressModal.show();
            } catch (e2) {
                // Approach 3: Direct attribute manipulation
                $('#training-progress-modal').addClass('show').css('display', 'block');
                $('body').addClass('modal-open').append('<div class="modal-backdrop fade show"></div>');
            }
        }
        
        // Reset charts
        if (window.lossChart) {
            window.lossChart.destroy();
        }
        if (window.perplexityChart) {
            window.perplexityChart.destroy();
        }
        
        // Initialize charts
        var lossCtx = document.getElementById('loss-chart').getContext('2d');
        window.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: true,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        var perplexityCtx = document.getElementById('perplexity-chart').getContext('2d');
        window.perplexityChart = new Chart(perplexityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Perplexity',
                        data: [],
                        borderColor: 'rgb(255, 159, 64)',
                        backgroundColor: 'rgba(255, 159, 64, 0.1)',
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'Validation Perplexity',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        fill: true,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Perplexity'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        // Load initial progress data
        $.ajax({
            url: '/api/training-sessions/' + sessionId + '/progress/',
            type: 'GET',
            success: function(response) {
                updateProgressUI(response);
                
                // If training is still running, connect to WebSocket for real-time updates
                if (response.session.status === 'Running') {
                    connectToTrainingWebSocket(sessionId);
                }
            },
            error: function() {
                alert('Error loading training progress data.');
            }
        });
    }
    
    // Connect to WebSocket for real-time training updates
    function connectToTrainingWebSocket(sessionId) {
        // Close existing connection if any
        if (window.trainingSocket) {
            window.trainingSocket.close();
        }
        
        // Create new WebSocket connection
        var wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        var wsUrl = wsProtocol + window.location.host + '/ws/training/' + sessionId + '/';
        
        window.trainingSocket = new WebSocket(wsUrl);
        
        window.trainingSocket.onmessage = function(e) {
            var data = JSON.parse(e.data);
            
            // Update progress bar
            $('#training-progress-bar').css('width', data.progress + '%');
            $('#training-progress-bar').attr('aria-valuenow', data.progress);
            $('#training-progress-bar').text(data.progress + '%');
            
            // Update status and message
            $('#training-status-text').text(data.status);
            $('#training-message-text').text(data.message);
            
            // Update charts if epoch data is available
            if (data.epoch !== undefined) {
                // Add data to loss chart
                window.lossChart.data.labels.push('Epoch ' + data.epoch);
                window.lossChart.data.datasets[0].data.push(data.train_loss);
                if (data.val_loss !== null) {
                    window.lossChart.data.datasets[1].data.push(data.val_loss);
                }
                window.lossChart.update();
                
                // Add data to perplexity chart
                window.perplexityChart.data.labels.push('Epoch ' + data.epoch);
                window.perplexityChart.data.datasets[0].data.push(data.train_perplexity);
                if (data.val_perplexity !== null) {
                    window.perplexityChart.data.datasets[1].data.push(data.val_perplexity);
                }
                window.perplexityChart.update();
            }
            
            // If training is completed or failed, close the WebSocket
            if (data.status === 'completed' || data.status === 'failed') {
                window.trainingSocket.close();
            }
        };
        
        window.trainingSocket.onclose = function() {
            console.log('Training WebSocket connection closed');
        };
    }
    
    // Update progress UI with data from API
    function updateProgressUI(data) {
        // Update session info
        $('#training-status-text').text(data.session.status);
        
        // Update progress bar based on status
        var progress = 0;
        if (data.session.status === 'Completed') {
            progress = 100;
        } else if (data.session.status === 'Failed') {
            progress = 0;
        } else if (data.progress.length > 0) {
            // Calculate progress based on epochs completed vs total epochs
            var lastEpoch = data.progress[data.progress.length - 1].epoch;
            var totalEpochs = 10; // Default to 10 if we don't know
            progress = Math.round((lastEpoch / totalEpochs) * 100);
        }
        
        $('#training-progress-bar').css('width', progress + '%');
        $('#training-progress-bar').attr('aria-valuenow', progress);
        $('#training-progress-bar').text(progress + '%');
        
        // Update charts
        if (data.progress.length > 0) {
            // Clear existing data
            window.lossChart.data.labels = [];
            window.lossChart.data.datasets[0].data = [];
            window.lossChart.data.datasets[1].data = [];
            window.perplexityChart.data.labels = [];
            window.perplexityChart.data.datasets[0].data = [];
            window.perplexityChart.data.datasets[1].data = [];
            
            // Add data points
            data.progress.forEach(function(point) {
                window.lossChart.data.labels.push('Epoch ' + point.epoch);
                window.lossChart.data.datasets[0].data.push(point.train_loss);
                if (point.val_loss !== null) {
                    window.lossChart.data.datasets[1].data.push(point.val_loss);
                }
                
                window.perplexityChart.data.labels.push('Epoch ' + point.epoch);
                window.perplexityChart.data.datasets[0].data.push(point.train_perplexity);
                if (point.val_perplexity !== null) {
                    window.perplexityChart.data.datasets[1].data.push(point.val_perplexity);
                }
            });
            
            // Update charts
            window.lossChart.update();
            window.perplexityChart.update();
        }
    }
    
    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
});
