import json
import os
import logging

# Set up logging configuration
# logging.basicConfig(level=logging.INFO)

class ConversationHistoryProcessor:
    """
    A class to manage conversation history for a torque-controlled robot interface.
    """

    def __init__(self, system_role_content, conversation_history_file):
        """
        Initializes the ConversationHistoryProcessor with a specific system role content 
        and a unique conversation history file.
        """
        self.system_role_content = system_role_content
        self.conversation_history_file = conversation_history_file
        self.ensure_history_file()

    def ensure_history_file(self):
        """
        Ensures that the conversation history file exists.
        """
        if not os.path.exists(self.conversation_history_file):
            os.makedirs(os.path.dirname(self.conversation_history_file), exist_ok=True)
            with open(self.conversation_history_file, 'w') as f:
                json.dump([], f)

    def get_recent_conversation_history(self):
        """
        Retrieves the recent conversation history along with the system role.
        """
        messages = []

        # Append system role to messages
        system_role = {
            "role": "system",
            "content": [{"type": "text", "text": self.system_role_content}]
        }
        messages.append(system_role)

        # Load conversation history
        try:
            with open(self.conversation_history_file, 'r') as f:
                data = json.load(f) or []
        except json.JSONDecodeError as e:
            logging.error(f"Error reading conversation history: {e}")
            data = []

        # Append last 10 items of data to messages
        messages.extend(data[-10:])

        return messages

    def update_conversation_history(self, transcription, response, image_url=None):
        """
        Updates the conversation history with a new transcription and response.
        If image_url is provided, it includes the image in the message.
        """
        # Get recent conversation history excluding the system role
        recent_conversation_history = self.get_recent_conversation_history()[1:]

        # Create new message entry
        content = [{"type": "text", "text": transcription}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        transcription_entry = {
            "role": "user",
            "content": content
        }
        response_entry = {
            "role": "system",
            "content": [{"type": "text", "text": response}]
        }

        # Append new messages to the conversation history
        recent_conversation_history.extend([transcription_entry, response_entry])

        # Save updated conversation history
        with open(self.conversation_history_file, 'w') as f:
            json.dump(recent_conversation_history, f)

    def reset_conversation_history(self):
        """
        Resets the conversation history by clearing the conversation history file.
        """
        with open(self.conversation_history_file, 'w') as f:
            json.dump([], f)
        logging.info(f"Conversation history in {self.conversation_history_file} has been reset.")
        return f"Conversation history in {self.conversation_history_file} has been reset."


if __name__ == "__main__":
    # Define different system roles
    role1 = (
        "You are an interface designed to adjust the stiffness matrix of a torque-controlled robot endpoint..."
    )
    role2 = (
        "You are a simplified assistant for adjusting stiffness matrices with basic operations..."
    )

    # Define unique conversation history files for each role
    history_file1 = "messages/conversation_history_role1.json"
    history_file2 = "messages/conversation_history_role2.json"

    # Create instances with different roles and conversation histories
    conv_manager1 = ConversationHistoryProcessor(system_role_content=role1, conversation_history_file=history_file1)
    conv_manager2 = ConversationHistoryProcessor(system_role_content=role2, conversation_history_file=history_file2)

    # Test updating conversation history
    transcription = "Adjust the stiffness matrix for the new task."
    response1 = "Role 1: Stiffness matrix adjusted with detailed analysis."
    response2 = "Role 2: Stiffness matrix adjusted simply."

    conv_manager1.update_conversation_history(transcription, response1)
    conv_manager2.update_conversation_history(transcription, response2)

    # Retrieve and print the recent conversation history for both instances
    recent_history1 = conv_manager1.get_recent_conversation_history()
    recent_history2 = conv_manager2.get_recent_conversation_history()

    print("History for Role 1:")
    print(json.dumps(recent_history1, indent=2))

    print("History for Role 2:")
    print(json.dumps(recent_history2, indent=2))

    # Reset the conversation history
    conv_manager1.reset_conversation_history()
    conv_manager2.reset_conversation_history()



"""
    CONVERSATION_HISTORY_FILE = "messages/conversation_history.json"
    SYSTEM_ROLE_CONTENT = (
        "You are an interface designed to adjust the stiffness matrix of a torque-controlled robot endpoint in a slide-in-the-groove position tracking task. The stiffness matrix defines a virtual 3D spring between the robot's actual endpoint position and the user's target position.\n\n"
        
        "Input data includes the user's commands in text form and, when relevant, an image URL of a mobile eye-tracker image that captures a screen displaying the current scene as captured by a camera at the robot endpoint with the user's gaze estimate as a red circle. The images are photos of a computer screen that displays the top-mounted camera feed on the robot's endpoint, capturing the teleoperation scene. Both textual and visual contexts inform your responses, with a maintained conversation history recording all user inputs and Your responses. Since the robot is equipped with torque-controlled motors, you can actively adjust arm stiffness based on voice and gaze data to achieve optimal task performance. Your primary task is to assist the user in adjusting the stiffness matrix based on voice and gaze input.\n\n"
        
        "**Systematic Approach to Determine the Stiffness Matrix:**\n\n"
        
        "1. **Analyze the Groove Orientation:**\n\n"
        
        "   - Examine the provided image to determine the groove's direction relative to the camera frame.\n"
        "   - **Camera Frame Definition:** The camera frame's X-axis points to the right, the Y-axis points away from you (depth), and the Z-axis points upwards.\n"
        "   - Identify the groove's angle and orientation with respect to the camera frame.\n\n"
        
        "2. **Define the Groove Coordinate System:**\n\n"
        
        "   - Establish a local coordinate system where the groove aligns with the X-axis.\n"
        "   - In this system, the groove direction is along the X-axis, and the perpendicular directions are along the Y and Z-axes.\n\n"
        
        "3. **Assign Stiffness Values in the Groove Frame:**\n\n"
        
        "   - **High Stiffness Along Groove Direction (X-axis):** Assign a high stiffness value (e.g., 1000 N/m).\n"
        "   - **Low Stiffness Perpendicular to Groove (Y and Z-axes):** Assign low stiffness values (e.g., 100 N/m).\n\n"
        
        "4. **Construct the Stiffness Matrix in the Groove Frame:**\n\n"
        
        "   - Create a diagonal stiffness matrix:\n\n"
        
        "     \\[\n"
        "     K_{\\text{groove}} = \\begin{bmatrix}\n"
        "     K_{\\text{high}} & 0 & 0 \\\\\n"
        "     0 & K_{\\text{low}} & 0 \\\\\n"
        "     0 & 0 & K_{\\text{low}}\n"
        "     \\end{bmatrix}\n"
        "     \\]\n\n"
        
        "5. **Determine the Rotation Matrix:**\n\n"
        
        "   - To transform the stiffness matrix from the groove frame to the camera frame, calculate the rotation matrix \( R \) that aligns the groove frame with the camera frame.\n"
        "   - **Steps to Calculate Rotation Matrix (3D Rotation):**\n\n"
        
        "     1. **Define Axis-Angle Rotation:** If you know the axis of rotation \\( \\mathbf{u} = [u_x, u_y, u_z] \\) and the angle \\( \\theta \\), use the axis-angle formula:\n"
        "        \\[\n"
        "        R = I + \\sin(\\theta) \\cdot [\\mathbf{u}]_\\times + (1 - \\cos(\\theta)) \\cdot ([\\mathbf{u}]_\\times)^2\n"
        "        \\]\n\n"
        
        "     2. **Quaternion Approach (Alternative):** If you know the orientation as a quaternion \\( q = (q_w, q_x, q_y, q_z) \\), convert it to a rotation matrix:\n"
        "        \\[\n"
        "        R = \\begin{bmatrix}\n"
        "        1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_z q_w) & 2(q_x q_z + q_y q_w) \\\\\n"
        "        2(q_x q_y + q_z q_w) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_x q_w) \\\\\n"
        "        2(q_x q_z - q_y q_w) & 2(q_y q_z + q_x q_w) & 1 - 2(q_x^2 + q_y^2)\n"
        "        \\end{bmatrix}\n"
        "        \\]\n\n"
        
        "   - Choose the approach based on available data (axis-angle or quaternion) and apply it to define the rotation from the groove frame to the camera frame.\n\n"
        
        "6. **Transform the Stiffness Matrix to the Camera Frame:**\n\n"
        
        "   - Use the rotation matrix to transform the stiffness matrix:\n"
        
        "     \\[\n"
        "     K_{\\text{camera}} = R \\cdot K_{\\text{groove}} \\cdot R^\\top\n"
        "     \\]\n\n"
        
        "   - Here, \( R^\\top \) is the transpose of the rotation matrix.\n\n"
        
        "7. **Compute the Final Stiffness Matrix:**\n\n"
        
        "   - Perform the matrix multiplication to obtain \\( K_{\\text{camera}} \\).\n"
        "   - This matrix represents the stiffness in the camera frame coordinate system.\n\n"
        
        "8. **Format the Output Correctly:**\n\n"
        
        "   - Present the stiffness matrix in the specified JSON format without any additional text or comments between the header and the code block.\n\n"
        
        "**Output Format:**\n\n"
        
        "To explain your findings, start your response using the exact format of the **Stiffness Matrix Analysis**. When presenting the stiffness matrix, output it as a JSON code block using the following exact format **without any additional text or comments** between the header and the code block:\n\n"
        
        "### Stiffness Matrix\n"
        "```json\n"
        "{\n"
        "  \"stiffness_matrix\": [\n"
        "    [X-X Value, Y-X Value, Z-X Value],\n"
        "    [X-Y Value, Y-Y Value, Z-Y Value],\n"
        "    [X-Z Value, Y-Z Value, Z-Z Value]\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        
        "**Formatting Guidelines:**\n"
        "- **Do not include any text, comments, or explanations between the \"### Stiffness Matrix\" header and the JSON code block.**\n"
        "- **Do not add comments or annotations within the JSON code block.**\n"
        "- **Only include numerical values in the stiffness matrix.**\n"
    )

"""
