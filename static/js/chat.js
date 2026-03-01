const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');

let isProcessing = false;

function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'You' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

async function sendMessage(content) {
    if (isProcessing) return;
    
    isProcessing = true;
    addMessage(content, 'user');
    messageInput.value = '';
    messageInput.disabled = true;
    
    showTyping();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: content })
        });
        
        const result = await response.json();
        hideTyping();
        addMessage(result.response, 'assistant');
    } catch (error) {
        console.error('Error:', error);
        hideTyping();
        addMessage('Error processing your request. Please try again.', 'assistant');
    }
    
    messageInput.disabled = false;
    messageInput.focus();
    isProcessing = false;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const content = messageInput.value.trim();
    if (content) {
        await sendMessage(content);
    }
});

messageInput.focus();
