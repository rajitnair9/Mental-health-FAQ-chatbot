
document.addEventListener("DOMContentLoaded", function(event) {
    // Selecting DOM elements
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("user-input");
    const chatbotMessages = document.getElementById("chatbot-messages");
    const sendBtn = document.getElementById("send-btn");
  
    //Event listener for the chat form submit
chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const userInput = chatInput.value;
      addUserMessage(userInput);
      sendUserMessage(userInput);
      chatInput.value = "";
      scrollToBottom();
    });
  
    //Event listener for the send button click
sendBtn.addEventListener("click", () => {
      const userInput = chatInput.value;
      addUserMessage(userInput);
      sendUserMessage(userInput);
      chatInput.value = "";
      scrollToBottom();
    });
  
    // Function to add a user message to the chat area
function addUserMessage(message) {
      const userMessageElement = `
        <div class="user-message">
          <p>${message}</p>
        </div>
      `;
      chatbotMessages.insertAdjacentHTML("beforeend", userMessageElement);
      scrollToBottom();
    }
  
    // Function to send user message to server and get response
function sendUserMessage(message) {
      showChatbotLoader();
      fetch("/get-response", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      })
        .then((response) => response.json())
        .then((data) => {
          const chatbotMessage = data.message;
          addChatbotMessage(chatbotMessage);
          hideChatbotLoader();
          scrollToBottom();
        })
        .catch((error) => {
          console.log("Error:", error);
          hideChatbotLoader();
        });
    }
  
    // Function to add a chatbot message to the chat area
function addChatbotMessage(message) {
      const chatbotMessageElement = `
        <div id="chatbot-message" class="chat-message">
          <p>${message}</p>
        </div>
      `;
      chatbotMessages.insertAdjacentHTML(
        "beforeend",
        chatbotMessageElement
      );
      scrollToBottom();
    }
  // Function to scroll to the bottom of the chat container

function scrollToBottom() {
    const scrollContainer = document.getElementById('chat-area');
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
  }
  
  
    // Function to hide the chatbot loader
function hideChatbotLoader() {
      const loaderElement = document.querySelector(".loader");
      if (loaderElement) {
        loaderElement.remove();
      }
    }
  
    // Add an event listener to the input field
    chatInput.addEventListener("keydown", function(event) {
      // Check if the key pressed is the enter key (key code 13)
      if (event.key === 'Enter') {
        // Prevent the default behavior of the enter key (submitting the form)
        event.preventDefault();

        // Trigger the click event on the send button
        document.getElementById("send-btn").click();
        scrollToBottom();
      }
    }); 

});

