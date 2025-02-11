document.addEventListener("DOMContentLoaded", function () {
  fetch("/get-states")
    .then((response) => response.json())
    .then((data) => {
      const job_list = document.getElementById("state_list");
      Object.entries(data).forEach(([state, id]) => {
        let option = document.createElement("option");
        option.value = id; // Set value as the integer ID
        option.text = state; // Set text as the state
        job_list.appendChild(option);
      });
    })
    .catch((error) => console.error("Error:", error));
});
