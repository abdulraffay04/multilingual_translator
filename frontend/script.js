const inputText = document.getElementById('inputText');
const outputText = document.getElementById('outputText');
const targetLang = document.getElementById('targetLang');
const translateBtn = document.getElementById('translateBtn');
const sourceLangLabel = document.getElementById('sourceLangLabel');
const statusMessage = document.getElementById('statusMessage');
const loader = document.getElementById('loader');

// Names map for your specific languages
const languageNames = {
    "ur": "Urdu",
    "en": "English",
    "fr": "French",
    "ar": "Arabic"
};

translateBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) {
        alert("Please enter text to translate.");
        return;
    }

    // 1. UI: Show Loading State
    loader.style.display = 'flex';
    translateBtn.disabled = true;
    statusMessage.innerText = "";
    outputText.value = ""; // Clear previous output

    try {
        const response = await fetch('http://127.0.0.1:5000/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                target_lang: targetLang.value
            })
        });

        const data = await response.json();

        if (response.ok) {
            // 2. Set Text Output
            outputText.value = data.translation;

            // 3. Handle Right-to-Left (RTL) for Urdu/Arabic
            const target = targetLang.value;
            if (target === "ur" || target === "ar") {
                outputText.style.direction = "rtl";
                outputText.style.fontFamily = "'Courier New', monospace"; // Or a specific Urdu font if installed
            } else {
                outputText.style.direction = "ltr";
                outputText.style.fontFamily = "inherit";
            }

            // 4. Update Detected Language Badge
            const detectedCode = data.detected_language;
            const readableName = languageNames[detectedCode] || detectedCode.toUpperCase();

            sourceLangLabel.innerHTML = `<i class="fa-solid fa-check"></i> Detected: ${readableName}`;
            sourceLangLabel.style.background = "#d1fae5";
            sourceLangLabel.style.color = "#065f46";

            statusMessage.innerText = "Success";
            statusMessage.style.color = "green";
        } else {
            statusMessage.innerText = "Error: " + data.error;
            statusMessage.style.color = "red";
        }

    } catch (error) {
        console.error("Error:", error);
        statusMessage.innerText = "Failed to connect to server.";
        statusMessage.style.color = "red";
    } finally {
        // 5. Hide Loading State
        loader.style.display = 'none';
        translateBtn.disabled = false;
    }
});