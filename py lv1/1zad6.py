def load_sms_collection(file_name):
    ham_messages = []
    spam_messages = []
    
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split('\t')
                label = parts[0]
                message = parts[1].strip()
                
                if label == 'ham':
                    ham_messages.append(message)
                elif label == 'spam':
                    spam_messages.append(message)
    
    except FileNotFoundError:
        print("Datoteka nije pronađena.")
        return None
    
    return ham_messages, spam_messages

def average_words_per_message(messages):
    total_words = sum(len(message.split()) for message in messages)
    num_messages = len(messages)
    
    if num_messages == 0:
        return 0
    
    return total_words / num_messages

def count_spam_messages_with_exclamation(messages):
    count = 0
    for message in messages:
        if message.endswith('!'):
            count += 1
    return count

def main():
    file_name = "SMSSpamCollection.txt" 
    
    ham_messages, spam_messages = load_sms_collection(file_name)
    
    if ham_messages is not None and spam_messages is not None:
        avg_words_ham = average_words_per_message(ham_messages)
        avg_words_spam = average_words_per_message(spam_messages)
        
        print("Prosječan broj riječi u porukama koje su tipa ham:", avg_words_ham)
        print("Prosječan broj riječi u porukama koje su tipa spam:", avg_words_spam)
        
        spam_with_exclamation = count_spam_messages_with_exclamation(spam_messages)
        print("Broj SMS poruka koje su tipa spam i završavaju uskličnikom:", spam_with_exclamation)

if __name__ == "__main__":
    main()
