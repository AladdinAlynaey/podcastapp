// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadProgress = document.getElementById('upload-progress');
const progressBar = document.getElementById('progress-bar');
const fileList = document.getElementById('file-list');
const noFileSelected = document.getElementById('no-file-selected');
const fileActions = document.getElementById('file-actions');
const selectedFileName = document.getElementById('selected-file-name');
const questionInput = document.getElementById('question-input');
const chatContainer = document.getElementById('chat-container');
const generatePodcastBtn = document.getElementById('generate-podcast-btn');
const numSpeakersSelect = document.getElementById('num-speakers');
const languageSelect = document.getElementById('language');
const podcastResult = document.getElementById('podcast-result');
const podcastScript = document.getElementById('podcast-script');
const podcastAudio = document.getElementById('podcast-audio');
const downloadAudioBtn = document.getElementById('download-audio-btn');
const scriptToggle = document.getElementById('script-toggle');
const scriptContent = document.getElementById('script-content');
const tabButtons = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// State
let selectedFile = null;
let conversationHistory = [];
let isTyping = false;
let currentAudioPath = null;
let scriptVisible = false;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
});

// Event Listeners
function setupEventListeners() {
  // File upload events
  fileInput.addEventListener('change', handleFileUpload);
  uploadArea.addEventListener('dragover', handleDragOver);
  uploadArea.addEventListener('dragleave', handleDragLeave);
  uploadArea.addEventListener('drop', handleFileDrop);

  // Chat events
  questionInput.addEventListener('keypress', handleQuestionInputKeypress);
  document.getElementById('chat-submit').addEventListener('click', askQuestion);

  // Tab events
  tabButtons.forEach(button => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
  });

  // Podcast events
  generatePodcastBtn.addEventListener('click', generatePodcast);
  downloadAudioBtn.addEventListener('click', downloadAudio);
  scriptToggle.addEventListener('click', toggleScriptVisibility);

  // Audio player events
  podcastAudio.addEventListener('timeupdate', updateAudioProgress);
  podcastAudio.addEventListener('loadedmetadata', initializeAudioPlayer);
}

// App Initialization
function initializeApp() {
  fetchFiles();
  addWelcomeMessage();
  setupEventListeners();
}

// Welcome Message
function addWelcomeMessage() {
  const welcomeMessage = `# Welcome to Alaadin's DocsPodcast! 👋

I can help you with:

* **Answering questions** about your uploaded documents
* **Generating podcast-style conversations** from your content
* **Supporting multiple languages** including English and Arabic

## To get started:
1. Upload a document using the upload area above
2. Select your document from the list
3. Ask questions or generate a podcast

Try uploading a PDF, Word, or text file to begin!`;

  appendBotMessage(welcomeMessage, true);
}

// File Handling
async function fetchFiles() {
  try {
    showLoading(fileList);

    const response = await fetch('/files');
    const data = await response.json();

    hideLoading(fileList);

    if (data.files && data.files.length > 0) {
      renderFiles(data.files);
    } else {
      showEmptyFileList();
    }
  } catch (error) {
    hideLoading(fileList);
    showError(fileList, "Error loading files. Please refresh the page.");
    console.error('Error fetching files:', error);
  }
}

function renderFiles(files) {
  fileList.innerHTML = '';

  files.forEach(file => {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';

    if (selectedFile === file.filename) {
      fileItem.classList.add('file-item-selected');
    }

    // Determine file type icon based on extension
    const fileExtension = file.filename.split('.').pop().toLowerCase();
    let fileIconClass = 'fa-file'; // Default icon

    // Map file extensions to specific icons
    switch (fileExtension) {
      case 'pdf':
        fileIconClass = 'fa-file-pdf';
        break;
      case 'doc':
      case 'docx':
        fileIconClass = 'fa-file-word';
        break;
      case 'xls':
      case 'xlsx':
        fileIconClass = 'fa-file-excel';
        break;
      case 'ppt':
      case 'pptx':
        fileIconClass = 'fa-file-powerpoint';
        break;
      case 'txt':
        fileIconClass = 'fa-file-alt';
        break;
      case 'csv':
        fileIconClass = 'fa-file-csv';
        break;
      case 'zip':
      case 'rar':
      case '7z':
        fileIconClass = 'fa-file-archive';
        break;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'svg':
        fileIconClass = 'fa-file-image';
        break;
      case 'mp3':
      case 'wav':
      case 'ogg':
        fileIconClass = 'fa-file-audio';
        break;
      case 'mp4':
      case 'avi':
      case 'mov':
        fileIconClass = 'fa-file-video';
        break;
      case 'html':
      case 'css':
      case 'js':
        fileIconClass = 'fa-file-code';
        break;
      default:
        fileIconClass = 'fa-file-alt';
    }

    // Format file size if available
    let fileSizeDisplay = '';
    if (file.size) {
      fileSizeDisplay = formatFileSize(file.size);
    }

    // Format upload date if available
    let uploadDateDisplay = '';
    if (file.uploaded_at) {
      uploadDateDisplay = formatDate(new Date(file.uploaded_at));
    }

    fileItem.innerHTML = `
      <div class="d-flex align-center">
        <i class="fas ${fileIconClass} file-icon"></i>
        <div class="file-info">
          <span class="file-name">${file.filename}</span>
          ${fileSizeDisplay || uploadDateDisplay ?
        `<div class="file-meta">
              ${fileSizeDisplay ? `<span class="file-size"><i class="fas fa-hdd"></i> ${fileSizeDisplay}</span>` : ''}
              ${uploadDateDisplay ? `<span class="file-date"><i class="fas fa-calendar-alt"></i> ${uploadDateDisplay}</span>` : ''}
            </div>` : ''}
        </div>
      </div>
      <div class="file-actions">
        <button class="btn btn-small btn-outline select-file-btn" data-filename="${file.filename}">
          <i class="fas fa-hand-pointer"></i>
          <span>Select</span>
        </button>
        <button class="btn btn-small btn-danger delete-file-btn" data-filename="${file.filename}">
          <i class="fas fa-trash-alt"></i>
        </button>
      </div>
    `;

    fileList.appendChild(fileItem);

    // Add event listeners
    fileItem.querySelector('.select-file-btn').addEventListener('click', () => {
      selectFile(file.filename);
    });

    fileItem.querySelector('.delete-file-btn').addEventListener('click', () => {
      deleteFile(file.filename);
    });
  });
}

function showEmptyFileList() {
  fileList.innerHTML = `
    <div class="text-center mt-2 mb-2">
      <p class="text-muted">No documents uploaded yet</p>
    </div>
  `;
}

function selectFile(filename) {
  selectedFile = filename;
  selectedFileName.textContent = filename;
  noFileSelected.classList.add('hidden');
  fileActions.classList.remove('hidden');

  // Update file list UI
  document.querySelectorAll('.file-item').forEach(item => {
    item.classList.remove('file-item-selected');
    if (item.querySelector('.file-name').textContent === filename) {
      item.classList.add('file-item-selected');
    }
  });

  // Reset chat with welcome message
  chatContainer.innerHTML = '';
  conversationHistory = [];

  // Add welcome message
  const welcomeMessage = `Hello! I'm your document assistant. Ask me any questions about "${filename}". I'll answer based only on the content of the document.`;
  appendBotMessage(welcomeMessage, true);

  // Reset podcast UI
  podcastResult.classList.add('hidden');
}

async function deleteFile(filename) {
  if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
    return;
  }

  try {
    showLoading(fileList);

    const response = await fetch('/delete-file', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ filename })
    });

    const data = await response.json();

    if (data.success) {
      // If the deleted file was selected, reset UI
      if (selectedFile === filename) {
        selectedFile = null;
        noFileSelected.classList.remove('hidden');
        fileActions.classList.add('hidden');
      }

      // Refresh file list
      fetchFiles();
    } else {
      showToast('error', `Error: ${data.error || 'Failed to delete file'}`);
    }
  } catch (error) {
    hideLoading(fileList);
    showToast('error', 'Error deleting file. Please try again.');
    console.error('Error deleting file:', error);
  }
}

async function handleFileUpload() {
  if (!fileInput.files || fileInput.files.length === 0) {
    return;
  }

  const file = fileInput.files[0];

  // Reset file input
  fileInput.value = '';

  // Validate file
  if (!validateFile(file)) {
    return;
  }

  // Create FormData object
  const formData = new FormData();
  formData.append('file', file);

  // Show progress bar
  uploadProgress.classList.remove('hidden');
  progressBar.style.width = '0%';

  try {
    const xhr = new XMLHttpRequest();

    // Setup progress event
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        const percentComplete = Math.round((event.loaded / event.total) * 100);
        progressBar.style.width = percentComplete + '%';
        progressBar.setAttribute('aria-valuenow', percentComplete);
      }
    });

    // Setup load event
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);

        if (response.success) {
          showToast('success', 'File uploaded successfully!');
          fetchFiles();
          // Use the server-returned filename (UUID-based for non-ASCII names)
          selectFile(response.filename || file.name);
        } else {
          showToast('error', `Error: ${response.error || 'Failed to upload file'}`);
        }
      } else {
        try {
          const errorResponse = JSON.parse(xhr.responseText);
          showToast('error', errorResponse.error || 'Error uploading file. Please try again.');
        } catch (e) {
          showToast('error', 'Error uploading file. Please try again.');
        }
      }

      // Hide progress bar
      uploadProgress.classList.add('hidden');
    });

    // Setup error event
    xhr.addEventListener('error', () => {
      showToast('error', 'Error uploading file. Please try again.');
      uploadProgress.classList.add('hidden');
    });

    // Send request
    xhr.open('POST', '/upload', true);
    xhr.send(formData);

  } catch (error) {
    uploadProgress.classList.add('hidden');
    showToast('error', 'Error uploading file. Please try again.');
    console.error('Error uploading file:', error);
  }
}

function validateFile(file) {
  // Check file size (max 0.5 MB = 512 KB)
  const MAX_SIZE = 0.5 * 1024 * 1024; // 512 KB
  if (file.size > MAX_SIZE) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    showToast('error', `File is too large (${sizeMB} MB). Maximum allowed size is 0.5 MB.`);
    return false;
  }

  // Check file type
  const acceptedTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'text/plain'
  ];

  if (!acceptedTypes.includes(file.type)) {
    showToast('error', 'Unsupported file type. Please upload PDF, DOCX, DOC, or TXT files.');
    return false;
  }

  return true;
}

function handleDragOver(e) {
  e.preventDefault();
  uploadArea.classList.add('upload-area-active');
}

function handleDragLeave() {
  uploadArea.classList.remove('upload-area-active');
}

function handleFileDrop(e) {
  e.preventDefault();
  uploadArea.classList.remove('upload-area-active');

  if (e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
    handleFileUpload();
  }
}

// Chat Functionality
function handleQuestionInputKeypress(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
}

async function askQuestion() {
  if (!selectedFile) {
    showToast('warning', 'Please select a document first.');
    return;
  }

  const question = questionInput.value.trim();
  if (!question) return;

  // Clear input
  questionInput.value = '';
  questionInput.disabled = true;
  document.getElementById('chat-submit').disabled = true;

  // Display user message
  appendUserMessage(question);

  // Show typing indicator
  showTypingIndicator();

  try {
    // Send question to API
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        filename: selectedFile,
        conversation_history: conversationHistory
      })
    });

    // Hide the typing indicator
    hideTypingIndicator();

    if (!response.ok) {
      const errorData = await response.json();
      appendBotMessage(`Sorry, there was an error: ${errorData.error || 'Unknown error'}`);
      return;
    }

    const data = await response.json();

    // Process the response to enhance markdown formatting
    let formattedAnswer = data.answer || "I'm sorry, I couldn't generate an answer.";

    // Ensure code blocks are properly formatted
    formattedAnswer = formattedAnswer.replace(/```([\s\S]*?)```/g, (match, p1) => {
      if (!p1.includes('\n')) {
        // This is an inline code block
        return '`' + p1 + '`';
      }

      // Check if language is specified
      const firstLine = p1.trim().split('\n')[0];
      const restContent = p1.trim().split('\n').slice(1).join('\n');

      if (firstLine && !firstLine.includes(' ') && firstLine.length < 20) {
        // Language specified, keep it
        return '```' + firstLine + '\n' + restContent + '\n```';
      } else {
        // No language specified, add default
        return '```text\n' + p1 + '\n```';
      }
    });

    // Ensure proper list formatting with spacing
    formattedAnswer = formattedAnswer.replace(/(\n[*-]\s.*\n)(?=[*-]\s)/g, '$1\n');

    // Add proper spacing before headings
    formattedAnswer = formattedAnswer.replace(/(\n)#{1,6}\s/g, '\n\n$&');

    // Add bot response to chat
    appendBotMessage(formattedAnswer, true);
  } catch (error) {
    hideTypingIndicator();
    appendBotMessage("Sorry, there was an error processing your question. Please try again.");
    console.error('Error asking question:', error);
  } finally {
    // Re-enable input
    questionInput.disabled = false;
    document.getElementById('chat-submit').disabled = false;
    questionInput.focus();
  }
}

// Helper function to ensure scrolling to bottom of chat
function scrollChatToBottom() {
  // Use a small timeout to ensure DOM has updated
  setTimeout(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }, 50);

  // Also try using scrollIntoView on the last message for better reliability
  const lastMessage = chatContainer.lastElementChild;
  if (lastMessage) {
    lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }
}

function appendUserMessage(text) {
  const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const messageElement = document.createElement('div');
  messageElement.className = 'chat-message user';
  messageElement.innerHTML = `
    <div class="message-bubble">
      <div class="message-content">${text}</div>
      <div class="message-timestamp">${timestamp}</div>
    </div>
    <div class="message-avatar">U</div>
  `;

  chatContainer.appendChild(messageElement);
  scrollChatToBottom();

  // Add to conversation history
  conversationHistory.push({ role: 'user', content: text });
}

function appendBotMessage(text, useMarkdown = false) {
  const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Process markdown if enabled
  const processedText = useMarkdown ? marked.parse(text) : text;

  // Sanitize HTML to prevent XSS
  const sanitizedText = DOMPurify.sanitize(processedText);

  // Create message element
  const messageElement = document.createElement('div');
  messageElement.className = 'chat-message bot';

  messageElement.innerHTML = `
    <div class="message-avatar bot-avatar">AI</div>
    <div class="message-bubble">
      <div class="message-content typing-animation"></div>
      <div class="message-timestamp">${timestamp}</div>
    </div>
  `;

  // Add to chat container
  chatContainer.appendChild(messageElement);

  // Scroll to the bottom
  scrollChatToBottom();

  // Get content element to type into
  const contentElement = messageElement.querySelector('.message-content');

  // Add to conversation history
  conversationHistory.push({ role: 'assistant', content: text });

  // Start typing effect with improved natural feel
  if (useMarkdown) {
    // Set the content first but make it invisible
    contentElement.innerHTML = sanitizedText;

    // Apply typewriter effect
    typeWriter(contentElement, sanitizedText);

    // Force remove the class after a maximum time to ensure the cursor disappears
    setTimeout(() => {
      contentElement.classList.remove('typing-animation');
    }, 12000);
  } else {
    // For plain text, still use typewriter for consistency
    typeWriter(contentElement, sanitizedText);

    // Force remove the class after a maximum time to ensure the cursor disappears
    setTimeout(() => {
      contentElement.classList.remove('typing-animation');
    }, 5000);
  }

  // Ensure scrolling works after typing is complete
  setTimeout(() => {
    scrollChatToBottom();
  }, 500);
}

// Enhanced typewriter animation that preserves HTML and adds natural typing variations
function typeWriter(element, html) {
  // Create a temporary container for the HTML
  const temp = document.createElement('div');
  temp.innerHTML = html;

  // Set empty content
  element.innerHTML = '';
  element.appendChild(temp);

  // Get all text nodes
  const allTextNodes = getAllTextNodes(temp);
  let visibleChars = 0;
  let totalChars = 0;

  // Count total characters for progress calculation
  allTextNodes.forEach(node => {
    totalChars += node.length;
  });

  // Hide all text initially
  hideAllText(temp);

  // Natural typing variation settings
  let typingSpeed = 20; // Base typing speed (ms) - made faster
  let nextCharDelay = 0;

  // Set a maximum typing time regardless of content length
  const maxTypingTime = 10000; // 10 seconds max
  const startTime = Date.now();

  // Start revealing characters with natural timing variations
  let typeInterval = setInterval(typeNextChars, typingSpeed);

  function typeNextChars() {
    // Check if max typing time has elapsed
    if (Date.now() - startTime > maxTypingTime) {
      // Force completion
      clearInterval(typeInterval);
      showAllText(temp);

      // Make sure to remove the animation class
      element.classList.remove('typing-animation');

      chatContainer.scrollTop = chatContainer.scrollHeight;
      return;
    }

    // Add natural variation to typing speed
    if (Math.random() < 0.05) { // Reduced probability for pauses
      // Occasional pause (thinking)
      nextCharDelay = 200 + Math.random() * 300; // Shorter pauses
    } else if (Math.random() < 0.2) {
      // Slight pause between words or sentences
      nextCharDelay = Math.random() * 50;
    } else {
      // Normal typing - increased speed
      nextCharDelay = 0;
      visibleChars += 3 + Math.floor(Math.random() * 5); // Show more characters at once
    }

    // Reveal characters up to the current count
    revealChars(temp, visibleChars);

    // Scroll to bottom as text appears
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Stop when all text is visible
    if (visibleChars >= totalChars) {
      clearInterval(typeInterval);

      // Make sure all text is visible 
      showAllText(temp);

      // Ensure the animation class is removed
      element.classList.remove('typing-animation');

      // Final scroll to ensure content is visible
      setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }, 100);

      return;
    }

    // Apply dynamic typing speed
    if (nextCharDelay > 0) {
      clearInterval(typeInterval);
      setTimeout(() => {
        typeInterval = setInterval(typeNextChars, typingSpeed);
      }, nextCharDelay);
    }
  }
}

// Helper functions for the typewriter effect
function getAllTextNodes(node) {
  const textNodes = [];
  const walker = document.createTreeWalker(
    node,
    NodeFilter.SHOW_TEXT,
    { acceptNode: (node) => node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT }
  );

  while (walker.nextNode()) {
    textNodes.push(walker.currentNode.textContent);
  }

  return textNodes;
}

function hideAllText(node) {
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);

  while (walker.nextNode()) {
    const textNode = walker.currentNode;
    const span = document.createElement('span');
    span.style.visibility = 'hidden';
    span.textContent = textNode.textContent;
    textNode.parentNode.replaceChild(span, textNode);
  }
}

function showAllText(node) {
  // Show all hidden text spans
  const hiddenSpans = node.querySelectorAll('span[style*="visibility: hidden"]');
  hiddenSpans.forEach(span => {
    span.style.visibility = 'visible';
  });
}

function revealChars(node, count) {
  let charCount = 0;

  // Find all spans with hidden text
  const hiddenSpans = node.querySelectorAll('span[style*="visibility: hidden"]');

  for (const span of hiddenSpans) {
    const text = span.textContent;

    // If we've already shown all characters in this function call
    if (charCount >= count) {
      break;
    }

    // If we'll show some or all of this span's characters
    if (charCount + text.length <= count) {
      // Show the entire span
      span.style.visibility = 'visible';
      charCount += text.length;
    } else {
      // Show part of this span
      const visiblePart = text.substring(0, count - charCount);
      const hiddenPart = text.substring(count - charCount);

      // Replace the span with two spans - one visible, one hidden
      const visibleSpan = document.createElement('span');
      visibleSpan.textContent = visiblePart;

      const hiddenSpan = document.createElement('span');
      hiddenSpan.style.visibility = 'hidden';
      hiddenSpan.textContent = hiddenPart;

      span.parentNode.insertBefore(visibleSpan, span);
      span.parentNode.insertBefore(hiddenSpan, span);
      span.parentNode.removeChild(span);

      charCount = count; // We've now shown exactly 'count' characters
      break;
    }
  }
}

function showTypingIndicator() {
  if (isTyping) return;

  isTyping = true;

  const typingElement = document.createElement('div');
  typingElement.className = 'chat-message bot typing-message';
  typingElement.innerHTML = `
    <div class="message-avatar bot-avatar">AI</div>
    <div class="typing-indicator">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;

  chatContainer.appendChild(typingElement);
  scrollChatToBottom();
}

function hideTypingIndicator() {
  const typingElement = document.querySelector('.typing-message');

  if (typingElement) {
    typingElement.remove();
  }

  isTyping = false;
}

// Podcast Functionality
async function generatePodcast() {
  if (!selectedFile) {
    showToast('warning', 'Please select a document first.');
    return;
  }

  // Get options
  const numSpeakers = numSpeakersSelect.value;
  const language = languageSelect.value;

  // Disable button and show loading state
  generatePodcastBtn.disabled = true;
  generatePodcastBtn.innerHTML = '<div class="loading"></div> Generating...';

  try {
    const response = await fetch('/generate-podcast', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        filename: selectedFile,
        num_speakers: parseInt(numSpeakers),
        language
      })
    });

    const data = await response.json();

    if (data.error) {
      showToast('error', `Error: ${data.error}`);
    }

    if (data.podcast_script) {
      // Show result
      podcastResult.classList.remove('hidden');

      // Set script content (use display_script if available, otherwise use podcast_script)
      const displayScript = data.display_script || data.podcast_script;
      podcastScript.innerHTML = DOMPurify.sanitize(marked.parse(displayScript));

      // Set audio source
      if (data.audio_path) {
        currentAudioPath = data.audio_path;
        podcastAudio.src = `${data.audio_path}?t=${Date.now()}`;
        podcastAudio.load();
      }

      // Show podcast section
      scrollToElement(podcastResult);
    } else {
      showToast('error', 'Failed to generate podcast. Please try again.');
    }
  } catch (error) {
    showToast('error', 'Error generating podcast. Please try again.');
    console.error('Error generating podcast:', error);
  } finally {
    // Re-enable button
    generatePodcastBtn.disabled = false;
    generatePodcastBtn.innerHTML = '<i class="fas fa-magic"></i> Generate Podcast';
  }
}

function downloadAudio() {
  if (currentAudioPath) {
    const link = document.createElement('a');
    link.href = currentAudioPath;
    link.download = `${selectedFile.replace(/\.[^/.]+$/, '')}_podcast.mp3`;
    link.click();
  }
}

function toggleScriptVisibility() {
  scriptVisible = !scriptVisible;

  if (scriptVisible) {
    scriptContent.style.maxHeight = scriptContent.scrollHeight + 'px';
    scriptToggle.innerHTML = '<i class="fas fa-chevron-up"></i>';
  } else {
    scriptContent.style.maxHeight = '0px';
    scriptToggle.innerHTML = '<i class="fas fa-chevron-down"></i>';
  }
}

// Audio Player
function initializeAudioPlayer() {
  const duration = podcastAudio.duration;
  document.querySelector('.time-total').textContent = formatTime(duration);
  document.querySelector('.time-current').textContent = '0:00';
  document.querySelector('.audio-progress-bar').style.width = '0%';
}

function updateAudioProgress() {
  const currentTime = podcastAudio.currentTime;
  const duration = podcastAudio.duration || 1;
  const progressPercent = (currentTime / duration) * 100;

  document.querySelector('.audio-progress-bar').style.width = `${progressPercent}%`;
  document.querySelector('.time-current').textContent = formatTime(currentTime);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}

// Tab Switching
function switchTab(tabId) {
  // Update active tab button
  tabButtons.forEach(button => {
    if (button.dataset.tab === tabId) {
      button.classList.add('active');
    } else {
      button.classList.remove('active');
    }
  });

  // Show active tab content
  tabContents.forEach(content => {
    if (content.id === tabId) {
      content.classList.remove('hidden');
    } else {
      content.classList.add('hidden');
    }
  });
}

// Utility Functions
function showLoading(element) {
  element.innerHTML = '<div class="text-center mt-2 mb-2"><div class="loading"></div></div>';
}

function hideLoading(element) {
  // This function should be implemented with the actual content
  // It will be overwritten by specific render functions
}

function showError(element, message) {
  element.innerHTML = `<div class="alert alert-danger mt-2 mb-2">${message}</div>`;
}

function scrollToElement(element) {
  element.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showToast(type, message) {
  // Create toast container if it doesn't exist
  let toastContainer = document.getElementById('toast-container');

  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.className = 'toast-container';
    document.body.appendChild(toastContainer);
  }

  // Create toast
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  // Set icon based on type
  let icon = 'info-circle';
  if (type === 'success') icon = 'check-circle';
  if (type === 'error') icon = 'exclamation-circle';
  if (type === 'warning') icon = 'exclamation-triangle';

  toast.innerHTML = `
    <div class="toast-icon">
      <i class="fas fa-${icon}"></i>
    </div>
    <div class="toast-content">${message}</div>
    <button class="toast-close">
      <i class="fas fa-times"></i>
    </button>
  `;

  // Add to container
  toastContainer.appendChild(toast);

  // Add event listener to close button
  toast.querySelector('.toast-close').addEventListener('click', () => {
    toast.classList.add('toast-hiding');
    setTimeout(() => {
      toast.remove();
    }, 300);
  });

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (toast.parentNode) {
      toast.classList.add('toast-hiding');
      setTimeout(() => {
        if (toast.parentNode) {
          toast.remove();
        }
      }, 300);
    }
  }, 5000);

  // Animate in
  setTimeout(() => {
    toast.classList.add('toast-visible');
  }, 10);
}

// Add toast styles dynamically
const toastStyles = document.createElement('style');
toastStyles.textContent = `
  .toast-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 350px;
  }
  
  .toast {
    display: flex;
    align-items: flex-start;
    padding: 15px;
    background-color: white;
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    transform: translateX(100%);
    opacity: 0;
    transition: transform 0.3s, opacity 0.3s;
    overflow: hidden;
    border-left: 4px solid;
  }
  
  .toast-visible {
    transform: translateX(0);
    opacity: 1;
  }
  
  .toast-hiding {
    transform: translateX(100%);
    opacity: 0;
  }
  
  .toast-icon {
    margin-right: 12px;
    font-size: 1.2rem;
    flex-shrink: 0;
  }
  
  .toast-content {
    flex: 1;
    margin-right: 12px;
  }
  
  .toast-close {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--gray-500);
    transition: color 0.2s;
    padding: 0;
    font-size: 0.9rem;
  }
  
  .toast-close:hover {
    color: var(--gray-800);
  }
  
  .toast-success {
    border-color: var(--success);
  }
  
  .toast-success .toast-icon {
    color: var(--success);
  }
  
  .toast-error {
    border-color: var(--danger);
  }
  
  .toast-error .toast-icon {
    color: var(--danger);
  }
  
  .toast-warning {
    border-color: var(--warning);
  }
  
  .toast-warning .toast-icon {
    color: var(--warning);
  }
  
  .toast-info {
    border-color: var(--primary);
  }
  
  .toast-info .toast-icon {
    color: var(--primary);
  }
  
  @media (max-width: 768px) {
    .toast-container {
      left: 20px;
      right: 20px;
      max-width: none;
    }
  }
`;

document.head.appendChild(toastStyles);

// Helper function to format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Helper function to format date
function formatDate(date) {
  const options = { year: 'numeric', month: 'short', day: 'numeric' };
  return date.toLocaleDateString(undefined, options);
} 