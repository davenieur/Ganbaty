$( document ).ready(function() {
    const csv_file = document.getElementById("ra_csvfile");

    csv_file.addEventListener("input", function (event) {
    if (csv_file.validity.typeMismatch) {
        csv_file.setCustomValidity("¡Se esperaba un archivo CSV!");
    } else {
        csv_file.setCustomValidity("");
    }
    });
    
    const support = document.getElementById("soporte");

    support.addEventListener("input", function (event) {
    if (support.validity.typeMismatch) {
        support.setCustomValidity("¡Se esperaba un número!");
    } else {
        support.setCustomValidity("");
    }
    });

 });
