# Battle Streamlit App

## Overview
The Battle Streamlit App is a 1 vs 1 battle simulation game where players can create their characters using custom prompts. The game utilizes OpenAI's language model to generate character stats and special skills based on the provided descriptions. Players can engage in battles, utilizing their characters' unique abilities to defeat their opponent.

## Project Structure
```
battle_streamlit_app
├── src
│   ├── app.py          # Main entry point for the Streamlit application
│   ├── battle.py       # Contains the Battle class for managing battle logic
│   ├── player.py       # Defines the Player class representing each player
│   ├── skill.py        # Contains the skill function for interpreting special skills
│   └── utils.py        # Utility functions for various tasks
├── requirements.txt     # Lists the dependencies required for the project
└── README.md            # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd battle_streamlit_app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage
- Upon launching the application, users will be prompted to enter descriptions for two characters.
- The application will generate character stats and skills based on the provided descriptions.
- Users can then simulate a battle between the two characters and view the results.

## Features
- Character creation based on user-defined prompts.
- Battle simulation with turn-based mechanics.
- Display of character stats and battle outcomes.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.