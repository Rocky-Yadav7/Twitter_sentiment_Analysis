function analyzeSentiment() {
    const text = document.getElementById("inputText").value;
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
      document.getElementById("result").innerText = `Sentiment: ${data.sentiment}`;
    })
    .catch(error => {
      console.error("Error:", error);
      document.getElementById("result").innerText = "An error occurred.";
    });
  }
  