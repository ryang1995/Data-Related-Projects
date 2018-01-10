import task1 as actor_tag
import task2 as genre_tag
import task3 as user_tag
import task4 as genre_diff

def main():
	while True:
		com_input = input()
		com_input_list = com_input.split()

		if len(com_input_list) == 3 and com_input_list[0] == 'print_actor_vector':
		    if com_input_list[2].upper() == 'TF':
		        actor_tag.main(com_input_list[1], 'TF')
		    elif com_input_list[2].upper() == 'TF-IDF':
		        actor_tag.main(com_input_list[1], 'TF-IDF')

		elif len(com_input_list) == 3 and com_input_list[0] == 'print_genre_vector':
		    if com_input_list[2].upper() == 'TF':
		        genre_tag.main(com_input_list[1], 'TF')
		    elif com_input_list[2].upper() == 'TF-IDF':
		        genre_tag.main(com_input_list[1], 'TF-IDF')
		    
		elif len(com_input_list) == 3 and com_input_list[0] == 'print_user_vector':
		    if com_input_list[2].upper() == 'TF':
				user_tag.main(com_input_list[1], 'TF')
		    elif com_input_list[2].upper() == 'TF-IDF':
		        user_tag.main(com_input_list[1], 'TF-IDF')
		    
		elif len(com_input_list) == 4 and com_input_list[0] == 'differentiate_genre':

		    if com_input_list[3].upper() == 'TF-IDF-DIFF':
		        genre_diff.main(com_input_list[1], com_input_list[2], com_input_list[3].upper())
		    elif com_input_list[3].upper() == 'P-DIFF1':
		        genre_diff.main(com_input_list[1], com_input_list[2], com_input_list[3].upper())
		    elif com_input_list[3].upper() == 'P-DIFF2':
		        genre_diff.main(com_input_list[1], com_input_list[2], com_input_list[3].upper())
		else:
			print "You input a wrong command."
if __name__ == '__main__':
	main()