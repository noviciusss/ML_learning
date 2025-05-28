console.log("Script.js version: 2.0 loaded"); // For checking if the new script is running

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorContainer = document.getElementById('errorContainer'); // Ensure this element exists in your index.html

    if (form) {
        form.addEventListener('submit', async function(event) {
            event.preventDefault();
            if (!validateForm()) {
                return;
            }

            const formData = new FormData(form);
            const data = Object.fromEntries(formData);

            if (loadingIndicator) {
                loadingIndicator.style.display = 'block';
            }
            if (errorContainer) {
                errorContainer.innerHTML = ''; // Clear previous errors
            }

            try {
                console.log("Sending data to /predict:", JSON.stringify(data));
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                console.log("Received response from server. Status:", response.status, "Ok:", response.ok);

                const responseText = await response.text(); // Get the response as text (HTML)
                console.log("Response text (first 100 chars):", responseText.substring(0, 100));


                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }

                if (response.ok) {
                    // Replace the current document's content with the HTML received from the server.
                    // This will effectively "render" the result.html page.
                    document.open();
                    document.write(responseText);
                    document.close();
                } else {
                    // Server returned an error status (e.g., 400, 500).
                    // The responseText likely contains an HTML error page from the server.
                    // We can display this, or a custom message.
                    // For now, let's try to display the server's error page if it's HTML,
                    // or a generic error if it's not what we expect.
                    console.error("Server returned an error status:", response.status);
                    // Attempt to display the server's HTML error page
                    if (responseText.toLowerCase().includes("<!doctype html") || responseText.toLowerCase().includes("<html>")) {
                        document.open();
                        document.write(responseText); // Display server's HTML error page
                        document.close();
                    } else {
                         displayError(`Server error: ${response.status} - ${response.statusText}. Response: ${responseText.substring(0,200)}`);
                    }
                }

            } catch (error) {
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
                console.error('Fetch or processing error:', error);
                console.error('Error name:', error.name);
                console.error('Error message:', error.message);
                console.error('Error stack:', error.stack);
                displayError('A network error or issue processing the response occurred. Please check the console for details. Message: ' + error.message);
            }
        });
    } else {
        console.error("Prediction form not found!");
    }


    function validateForm() {
        if (!form) return false;
        const inputs = form.querySelectorAll('input[required], select[required]');
        let isValid = true;
        let firstInvalidElement = null;

        inputs.forEach(input => {
            // Clear previous custom validation messages
            const existingFeedback = input.parentNode.querySelector('.custom-invalid-feedback');
            if (existingFeedback) {
                existingFeedback.remove();
            }
            input.classList.remove('is-invalid');

            if (!input.value || !input.value.trim()) {
                input.classList.add('is-invalid');
                const feedback = document.createElement('div');
                feedback.classList.add('invalid-feedback', 'custom-invalid-feedback'); // Bootstrap class + custom class
                feedback.textContent = 'This field is required.';
                // Insert after the input, or adjust if using Bootstrap's form-floating or input-group
                input.parentNode.insertBefore(feedback, input.nextSibling);
                isValid = false;
                if (!firstInvalidElement) {
                    firstInvalidElement = input;
                }
            }
        });

        if (firstInvalidElement) {
            firstInvalidElement.focus();
        }
        return isValid;
    }

    function displayError(errorMessage) {
        console.error('Displaying Error:', errorMessage);
        if (errorContainer) {
            errorContainer.innerHTML = `<div class="alert alert-danger" role="alert">${errorMessage}</div>`;
        } else {
            alert(errorMessage); // Fallback
        }
    }

    // displayResult is not directly used if the whole page is replaced,
    // but kept for potential future use or if error handling needs to update index.html
    function displayResult(resultData) {
        console.log('displayResult called (not typically used with full page replacement):', resultData);
    }
});