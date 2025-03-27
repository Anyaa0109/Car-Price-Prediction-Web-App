function predictPrice() {
    let year = document.getElementById("year").value;
    let present_price = document.getElementById("present_price").value;
    let kms_driven = document.getElementById("kms_driven").value;
    let fuel_type = document.getElementById("fuel_type").value;
    let seller_type = document.getElementById("seller_type").value;
    let transmission = document.getElementById("transmission").value;
    let owner = document.getElementById("owner").value;

    if (!year || !present_price || !kms_driven || !owner) {
        alert("Please fill all required fields.");
        return;
    }

    let data = {
        "year": parseInt(year),
        "present_price": parseFloat(present_price),
        "kms_driven": parseInt(kms_driven),
        "fuel_type": parseInt(fuel_type),
        "seller_type": parseInt(seller_type),
        "transmission": parseInt(transmission),
        "owner": parseInt(owner)
    };

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("prediction-result").innerHTML = "Predicted Price: ₹" + data.predicted_price.toFixed(2);
    })
    .catch(error => console.error('Error:', error));
}
