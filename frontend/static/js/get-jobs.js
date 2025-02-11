document.addEventListener("DOMContentLoaded", function () {
  fetch("/get-jobs")
    .then((response) => response.json())
    .then((data) => {
      const job_list = document.getElementById("job_list");
      Object.entries(data).forEach(([job, id]) => {
        let option = document.createElement("option");
        option.value = id; // Set value as the integer ID
        option.text = job; // Set text as the job title
        job_list.appendChild(option);
      });
    })
    .catch((error) => console.error("Error:", error));
});
