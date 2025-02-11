document
  .getElementById("prediction-form")
  .addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent form submission

    const job = document.getElementById("job_list").value;
    const state = document.getElementById("state_list").value;
    const experience = document.getElementById("experience_list").value;
    const work_type = document.getElementById("work_type_list").value;

    // Send POST request to Flask backend
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        features: [
          parseFloat(job),
          parseFloat(state),
          parseFloat(experience),
          parseFloat(work_type),
        ], // Send feature data
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
