document
  .getElementById("prediction-form")
  .addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent form submission

    const state = document.getElementById("state_list").value;
    const job = document.getElementById("job_list").value;
    const experience = document.getElementById("experience_list").value;

    // Send POST request to Flask backend
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        features: [parseFloat(state), parseFloat(job), parseFloat(experience)], // Send feature data
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display prediction result
        if (data.prediction) {
          document.getElementById(
            "prediction-result"
          ).innerText = `Prediction: $${data.prediction}`;
        } else if (data.error) {
          document.getElementById(
            "prediction-result"
          ).innerText = `Error: ${data.error}`;
        }
      })
      .catch((error) => console.error("Error:", error));
  });
