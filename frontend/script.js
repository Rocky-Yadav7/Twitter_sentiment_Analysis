function analyzeSentiment() {
  const text = document.getElementById("inputText").value.trim();
  const resultElement = document.getElementById("result");
  
  if (!text) {
    resultElement.innerText = "Please enter some text to analyze";
    return;
  }

  resultElement.innerText = "Analyzing...";
  
  fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  })
  .then(response => {
    if (!response.ok) {
      return response.json().then(err => { throw new Error(err.error || "Server error"); });
    }
    return response.json();
  })
  .then(data => {
    const confidencePercent = (data.confidence * 100).toFixed(1);
    resultElement.innerHTML = `
      <strong>Sentiment:</strong> ${data.sentiment}<br>
      <strong>Confidence:</strong> ${confidencePercent}%
    `;
  })
  .catch(error => {
    console.error("Error:", error);
    resultElement.innerText = "Error: " + error.message;
  });
}