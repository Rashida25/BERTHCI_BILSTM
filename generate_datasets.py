"""
BERTHCI Dataset Generator
Generates synthetic HCI prompt datasets for training and testing
"""

import pandas as pd
import numpy as np
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# =========================================================================
# SYNTHETIC DATASET GENERATOR
# =========================================================================
class SyntheticHCIDatasetGenerator:
    """Generate realistic synthetic HCI prompt dataset"""
    
    def __init__(self, num_samples=5000, noise_level=0.15):
        self.num_samples = num_samples
        self.noise_level = noise_level
        
        self.categories = {
            "Real-Time Translation UI": {
                "templates": [
                    "Translate {phrase} to {language}",
                    "How do you say {phrase} in {language}",
                    "Convert {phrase} from English to {language}",
                    "{phrase} in {language}",
                    "What's the {language} word for {phrase}",
                    "Can you translate {phrase} into {language}"
                ],
                "phrases": [
                    "hello", "goodbye", "thank you", "how are you", "good morning",
                    "please help me", "where is the bathroom", "nice to meet you",
                    "I love you", "excuse me", "sorry", "yes", "no", "maybe"
                ],
                "languages": [
                    "Spanish", "French", "German", "Italian", "Japanese", "Chinese",
                    "Korean", "Arabic", "Russian", "Portuguese", "Hindi", "Dutch"
                ]
            },
            
            "Voice Calculator": {
                "templates": [
                    "Calculate {expr}",
                    "What is {expr}",
                    "Compute {expr}",
                    "Solve {expr}",
                    "{expr} equals what",
                    "Find the result of {expr}",
                    "Give me {expr}"
                ],
                "operations": [
                    "{a} plus {b}", "{a} minus {b}", "{a} times {b}", "{a} divided by {b}",
                    "{a} add {b}", "{a} subtract {b}", "square root of {a}", "{a} squared",
                    "{a} multiplied by {b}", "{a} over {b}", "{a} to the power of {b}"
                ]
            },
            
            "Location Services UI": {
                "templates": [
                    "Where is {place}",
                    "What is the capital of {country}",
                    "Capital of {country}",
                    "Navigate to {place}",
                    "Find {place} near me",
                    "Show me {place}",
                    "Directions to {place}",
                    "How far is {place}"
                ],
                "places": [
                    "New York", "Paris", "Tokyo", "the nearest hospital", "the library",
                    "downtown", "the airport", "Central Park", "the museum", "the mall",
                    "Times Square", "the beach", "the train station"
                ],
                "countries": [
                    "France", "Germany", "Japan", "India", "Brazil", "Canada", "Italy",
                    "Spain", "Australia", "Mexico", "China", "Russia"
                ]
            },
            
            "Educational Bot": {
                "templates": [
                    "Explain {concept}",
                    "What is {concept}",
                    "Define {concept}",
                    "How does {concept} work",
                    "Describe {concept}",
                    "Tell me about {concept}",
                    "Teach me {concept}",
                    "I need to understand {concept}"
                ],
                "concepts": [
                    "photosynthesis", "gravity", "democracy", "machine learning",
                    "evolution", "blockchain", "DNA", "quantum mechanics", "AI",
                    "climate change", "the water cycle", "electricity", "magnetism",
                    "cell division", "the solar system", "enzymes"
                ]
            },
            
            "Knowledge Assistant": {
                "templates": [
                    "Who wrote {title}",
                    "Who invented {invention}",
                    "When was {event}",
                    "History of {subject}",
                    "Tell me about {event}",
                    "What happened in {event}",
                    "Who discovered {invention}",
                    "Facts about {subject}"
                ],
                "titles": [
                    "Romeo and Juliet", "1984", "The Great Gatsby", "Harry Potter",
                    "To Kill a Mockingbird", "Pride and Prejudice", "The Odyssey"
                ],
                "inventions": [
                    "the telephone", "the light bulb", "the airplane", "the computer",
                    "the internet", "penicillin", "the steam engine", "the radio"
                ],
                "events": [
                    "World War II", "the Renaissance", "the Industrial Revolution",
                    "the French Revolution", "the Moon landing", "the Cold War"
                ],
                "subjects": [
                    "ancient Egypt", "dinosaurs", "the Roman Empire", "space exploration",
                    "the pyramids", "Greek mythology", "the Titanic"
                ]
            },
            
            "Chatbot Interface": {
                "templates": [
                    "Tell me a {type}",
                    "Can you help me",
                    "Give me advice on {topic}",
                    "Recommend {item}",
                    "I would like to {action}",
                    "Suggest something",
                    "Help me decide",
                    "What do you think about {topic}",
                    "Share a {type} with me",
                    "I need suggestions for {item}"
                ],
                "types": ["joke", "story", "fun fact", "quote", "riddle"],
                "topics": ["career", "relationships", "health", "education", "hobbies"],
                "items": ["a book", "a movie", "a restaurant", "a game", "music"],
                "actions": ["learn something new", "relax", "exercise", "cook"]
            },
            
            "Conversational UI": {
                "templates": [
                    "How are you",
                    "What can you do",
                    "Hello",
                    "Good morning",
                    "Thank you",
                    "Goodbye",
                    "What is your name",
                    "Can you help me",
                    "Tell me something",
                    "I'm confused",
                    "That's interesting",
                    "Hi there",
                    "Hey",
                    "Good afternoon",
                    "Nice to meet you",
                    "How's it going",
                    "What's up"
                ]
            },

            "Code Generator": {
                "templates": [
                    "Write {lang} code for {task}",
                    "Generate code in {lang} to {task}",
                    "Create a {lang} script that {task}",
                    "How to {task} in {lang}",
                    "Code example for {task} in {lang}"
                ],
                "langs": ["Python", "JavaScript", "HTML", "CSS"],
                "tasks": [
                    "a login API",
                    "sort a list",
                    "create a web page",
                    "style a button",
                    "fetch data from API",
                    "validate form",
                    "animate an element",
                    "compute factorial",
                    "parse JSON",
                    "handle errors"
                ]
            },

            "Code Rectification": {
                "templates": [
                    "Fix this code: {code_snippet}",
                    "Debug the following: {code_snippet}",
                    "There's an error in {code_snippet}",
                    "Correct this {lang} code: {code_snippet}"
                ],
                "code_snippets": [
                    "if x = 10: print('hi')",
                    "for i in range(10) print(i)",
                    "<div class='btn' onclick='alert('hi')'>",
                    "button { color red; }",
                    "fetch(url).then(res => res.json",
                    "try: pass except: pass"
                ],
                "langs": ["Python", "JavaScript", "HTML", "CSS"]
            },

            "Graph Generation": {
                "templates": [
                    "Plot a graph for {data_type}",
                    "Visualize {data_type} as {graph_type}",
                    "Create a chart showing {data_type}",
                    "Graph {data_type}"
                ],
                "data_types": ["user growth", "sales data", "temperature changes", "stock prices", "population growth"],
                "graph_types": ["line", "bar", "pie", "scatter"]
            }
        }
        
        # Ambiguous templates for realistic difficulty
        self.ambiguous_templates = [
            "Show me information about {topic}",
            "Find the answer to {question}",
            "I need help with {task}",
            "Tell me more",
            "What about {topic}",
            "Help with {task}",
            "Information on {topic}",
        ]
    
    def generate_prompt(self, category):
        """Generate a single prompt for the given category"""
        config = self.categories[category]
        
        # 10% chance to use ambiguous template
        if random.random() < 0.10:
            template = random.choice(self.ambiguous_templates)
            topics = ["science", "history", "math", "geography", "technology"]
            tasks = ["homework", "research", "learning", "understanding"]
            questions = ["this problem", "that question", "my query"]
            
            if "{topic}" in template:
                return template.format(topic=random.choice(topics))
            elif "{task}" in template:
                return template.format(task=random.choice(tasks))
            elif "{question}" in template:
                return template.format(question=random.choice(questions))
            return template
        
        template = random.choice(config["templates"])
        
        if category == "Real-Time Translation UI":
            return template.format(
                phrase=random.choice(config["phrases"]),
                language=random.choice(config["languages"])
            )
        
        elif category == "Voice Calculator":
            a, b = random.randint(1, 100), random.randint(1, 50)
            operation = random.choice(config["operations"]).format(a=a, b=b)
            return template.format(expr=operation)
        
        elif category == "Location Services UI":
            if "{place}" in template:
                return template.format(place=random.choice(config["places"]))
            return template.format(country=random.choice(config["countries"]))
        
        elif category == "Educational Bot":
            return template.format(concept=random.choice(config["concepts"]))
        
        elif category == "Knowledge Assistant":
            if "{title}" in template:
                return template.format(title=random.choice(config["titles"]))
            elif "{invention}" in template:
                return template.format(invention=random.choice(config["inventions"]))
            elif "{event}" in template:
                return template.format(event=random.choice(config["events"]))
            elif "{subject}" in template:
                return template.format(subject=random.choice(config["subjects"]))
            return template.format(subject="science")
        
        elif category == "Chatbot Interface":
            if "{type}" in template:
                return template.format(type=random.choice(config["types"]))
            elif "{topic}" in template:
                return template.format(topic=random.choice(config["topics"]))
            elif "{item}" in template:
                return template.format(item=random.choice(config["items"]))
            elif "{action}" in template:
                return template.format(action=random.choice(config["actions"]))
            return template
        
        elif category == "Conversational UI":
            return template
        
        elif category == "Code Generator":
            lang = random.choice(config["langs"])
            task = random.choice(config["tasks"])
            return template.format(lang=lang, task=task)
        
        elif category == "Code Rectification":
            code_snippet = random.choice(config["code_snippets"])
            if "{lang}" in template:
                lang = random.choice(config["langs"])
                return template.format(lang=lang, code_snippet=code_snippet)
            return template.format(code_snippet=code_snippet)
        
        elif category == "Graph Generation":
            data_type = random.choice(config["data_types"])
            if "{graph_type}" in template:
                graph_type = random.choice(config["graph_types"])
                return template.format(data_type=data_type, graph_type=graph_type)
            return template.format(data_type=data_type)
    
    def determine_prompt_type(self, prompt):
        """Classify prompt type based on structure"""
        prompt_lower = prompt.lower()
        
        if any(prompt_lower.startswith(q) for q in 
               ["what", "where", "when", "who", "how", "why"]):
            return "Interrogative"
        
        if any(prompt_lower.startswith(c) for c in 
               ["calculate", "translate", "show", "find", "explain", "solve"]):
            return "Imperative"
        
        if any(prompt_lower.startswith(c) for c in 
               ["hello", "hi", "good morning", "thank you", "hey"]):
            return "Conversational"
        
        return "Declarative"
    
    def add_noise(self, prompt):
        """Add variations to make dataset more realistic"""
        variations = [
            lambda p: p + "?",
            lambda p: "Please " + p,
            lambda p: p + " please",
            lambda p: p.lower(),
            lambda p: p + "!!",
            lambda p: "Could you " + p,
            lambda p: p.replace("the", "a"),
            lambda p: p + " thanks",
            lambda p: p.capitalize()
        ]
        return random.choice(variations)(prompt)
    
    def generate_dataset(self, dataset_type="training"):
        """Generate complete dataset"""
        print("\n" + "="*70)
        print(f" GENERATING {dataset_type.upper()} DATASET")
        print("="*70)
        
        data = []
        samples_per_category = self.num_samples // len(self.categories)
        
        # Use different random seed for training vs testing
        seed_offset = 0 if dataset_type == "training" else 1000
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)
        
        for category in self.categories.keys():
            print(f"Generating {samples_per_category} samples for: {category}")
            
            for _ in range(samples_per_category):
                prompt = self.generate_prompt(category)
                
                # Add noise/variation
                if random.random() < self.noise_level:
                    prompt = self.add_noise(prompt)
                
                data.append({
                    "Prompt": prompt,
                    "Prompt_Type": self.determine_prompt_type(prompt),
                    "Prompt_Length": len(prompt),
                    "HCI_Application": category
                })
        
        # Fill remaining samples to reach exact count
        remaining = self.num_samples - len(data)
        for _ in range(remaining):
            category = random.choice(list(self.categories.keys()))
            prompt = self.generate_prompt(category)
            
            if random.random() < self.noise_level:
                prompt = self.add_noise(prompt)
            
            data.append({
                "Prompt": prompt,
                "Prompt_Type": self.determine_prompt_type(prompt),
                "Prompt_Length": len(prompt),
                "HCI_Application": category
            })
        
        # Reset seeds
        random.seed(42)
        np.random.seed(42)
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42 + seed_offset).reset_index(drop=True)
        
        print(f"\n✓ Generated {len(df):,} total records")
        print("\nClass Distribution:")
        print(df["HCI_Application"].value_counts())
        print("\nPrompt Type Distribution:")
        print(df["Prompt_Type"].value_counts())
        
        return df

# =========================================================================
# MAIN EXECUTION
# =========================================================================
def main():
    print("\n" + "="*70)
    print(" BERTHCI DATASET GENERATION")
    print("="*70)
    
    # Create CSV directory if it doesn't exist
    os.makedirs("./CSV", exist_ok=True)
    
    # Define paths
    train_path = "./CSV/prompt_engineering_dataset_train.csv"
    test_path = "./CSV/prompt_engineering_dataset_test.csv"
    
    # Check if datasets already exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        response = input("\nDatasets already exist. Regenerate? (y/n): ")
        if response.lower() != 'y':
            print("✓ Using existing datasets")
            return
    
    # Generate training dataset (5000 samples)
    print("\n[1/2] Generating Training Dataset...")
    generator_train = SyntheticHCIDatasetGenerator(num_samples=5000, noise_level=0.15)
    train_df = generator_train.generate_dataset(dataset_type="training")
    
    # Generate test dataset (1500 samples)
    print("\n[2/2] Generating Test Dataset...")
    generator_test = SyntheticHCIDatasetGenerator(num_samples=1500, noise_level=0.15)
    test_df = generator_test.generate_dataset(dataset_type="testing")
    
    # Save datasets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\n" + "="*70)
    print(" DATASET GENERATION COMPLETED")
    print("="*70)
    print(f"\n✓ Training dataset saved: {train_path}")
    print(f"  - Total samples: {len(train_df):,}")
    print(f"  - Categories: {train_df['HCI_Application'].nunique()}")
    
    print(f"\n✓ Test dataset saved: {test_path}")
    print(f"  - Total samples: {len(test_df):,}")
    print(f"  - Categories: {test_df['HCI_Application'].nunique()}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python train_model.py")
    print("2. Run: python app.py")
    print("3. Open: http://localhost:5000")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()