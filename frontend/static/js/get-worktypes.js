document.addEventListener("DOMContentLoaded", function () {
  fetch("/get-worktypes")
    .then((response) => response.json())
    .then((data) => {
      const job_list = document.getElementById("work_type_list");
      Object.entries(data).forEach(([type, id]) => {
        let option = document.createElement("option");
        option.value = id; // Set value as the integer ID
        option.text = type; // Set text as the work type
        job_list.appendChild(option);
      });
    })
    .catch((error) => console.error("Error:", error));
});
