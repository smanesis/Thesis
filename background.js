chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "summarize") {
    fetch('http://localhost:5000/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: request.url })
    })
    .then(response => response.json())
    .then(data => {
      chrome.runtime.sendMessage({ action: "displaySummary", summary: data.summary });
    });
  }
});
