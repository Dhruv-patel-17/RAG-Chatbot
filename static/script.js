document.addEventListener("DOMContentLoaded", function () {

    /* ---------- ELEMENTS ---------- */
    const queryTypeSelect = document.getElementById("query_type");
    const topicDiv = document.getElementById("topic_div");
    const customDiv = document.getElementById("custom_div");

    const form = document.querySelector("form");
    const loader = document.getElementById("loader");
    const text = document.getElementById("loadingText");
    const success = document.getElementById("successCheck");


    /* ---------- LOADING STEPS ---------- */
    const steps = [
        "Retrieving context",
        "Analyzing PYQs",
        "Selecting relevant questions",
        "Generating questions",
        "Formatting paper"
    ];

    let i = 0;
    let interval;

    /* ---------- FORM SUBMIT ---------- */
    form.addEventListener("submit", () => {

        /* mark that generation started */
        localStorage.setItem("generated", "true");

        /* remove previous output smoothly */
        const oldOutput = document.querySelector(".output");
        if (oldOutput) {
            oldOutput.style.opacity = "0";
            setTimeout(() => oldOutput.remove(), 300);
        }

        /* show loader */
        loader.style.display = "flex";

        /* reset steps */
        i = 0;
        text.textContent = steps[i];

        /* animated step text */
        interval = setInterval(() => {
            i = (i + 1) % steps.length;
            text.textContent = steps[i];
        }, 1500);
    });

    /* ---------- PAGE LOAD AFTER RESPONSE ---------- */
    window.addEventListener("load", () => {

        loader.style.display = "none";

        if (localStorage.getItem("generated") === "true") {

            clearInterval(interval);

            success.style.display = "block";

            localStorage.removeItem("generated");

            setTimeout(() => {
                success.style.opacity = "0";
                setTimeout(() => {
                    success.style.display = "none";
                    success.style.opacity = "1";
                }, 400);
            }, 1200);
        }
    });
    document.getElementById("downloadBtn").addEventListener("click", function () {
        this.textContent = "Preparing PDF...";
        this.disabled = true;

        setTimeout(() => {
            window.location.href = "/download_pdf";
            this.textContent = "Download PDF";
            this.disabled = false;
        }, 800);
    });

    /* ---------- DROPDOWN SWITCH ---------- */
    queryTypeSelect.addEventListener("change", function () {

        const value = this.value;

        topicDiv.style.display = "none";
        customDiv.style.display = "none";

        if (value === "topic")
            topicDiv.style.display = "block";

        else if (value === "custom")
            customDiv.style.display = "block";
    });


});