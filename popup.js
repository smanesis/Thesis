document.getElementById('summarizeButton').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    let currentUrl = tabs[0].url;
    chrome.runtime.sendMessage({ action: "summarize", url: currentUrl });
  });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "displaySummary") {
    document.getElementById('summary').textContent = request.summary;
  }
});
