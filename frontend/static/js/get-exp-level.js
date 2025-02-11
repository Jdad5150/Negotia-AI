document.addEventListener("DOMContentLoaded", function () {
  fetch("/get-exp-level")
    .then((response) => response.json())
    .then((data) => {
      const job_list = document.getElementById("experience_list");
      Object.entries(data).forEach(([level, id]) => {
        let option = document.createElement("option");
        option.value = id; // Set value as the integer ID
        option.text = level; // Set text as the work type
        job_list.appendChild(option);
      });
    })
    .catch((error) => console.error("Error:", error));
});
