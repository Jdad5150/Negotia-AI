document
  .getElementById("prediction-form")
  .addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent form submission

    // Get selected values from dropdowns
    const state = document.getElementById("state_list");
    const state_id = state.options[state.selectedIndex].value;
    const state_title = state.options[state.selectedIndex].text;

    const job = document.getElementById("job_list");
    const job_id = job.options[job.selectedIndex].value;
    const job_title = job.options[job.selectedIndex].text;

    const experience = document.getElementById("experience_list");
    const experience_id = experience.options[experience.selectedIndex].value;
    const experience_title = experience.options[experience.selectedIndex].text;

    const resultElement = document.getElementById("prediction-result");
    const titleElement = document.getElementById("response-container-h1");
    const subtitleElement = document.getElementById("response-subtitle");

    // Send POST request to Flask backend
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        features: [
          parseFloat(state_id),
          parseFloat(job_id),
          parseFloat(experience_id),
        ],
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display prediction result
        if (data.prediction) {
          const roundedPrediction = Math.round(data.prediction).toLocaleString(
            "en-US",
            { style: "currency", currency: "USD" }
          );
          const upperBound = (
            Math.round((Math.round(data.prediction) * 1.1) / 10000) * 10000
          ).toLocaleString("en-US", { style: "currency", currency: "USD" });
          const lowerBound = (
            Math.round((Math.round(data.prediction) * 0.9) / 10000) * 10000
          ).toLocaleString("en-US", { style: "currency", currency: "USD" });

          resultElement.innerText = `${experience_title} ${job_title} in ${state_title} should make ${roundedPrediction} per year.\n\nStart your negotiation with a salary of ${upperBound} and take no less than ${lowerBound}.`;
          titleElement.style.display = "none";
          subtitleElement.style.display = "none";
        } else if (data.error) {
          resultElement.innerText = `Error: ${data.error}`;
        }
      })
      .catch((error) => console.error("Error:", error));
  });
