document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Get the ticker symbol entered by the user
    const tickerSymbol = document.getElementById('tickerSymbol').value;

    // Construct the request body
    const requestBody = {
        ticker: tickerSymbol
    };

    // Send POST request to /predict endpoint
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result on the webpage
        document.getElementById('predictionResult').innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => console.error('Error:', error));
});
