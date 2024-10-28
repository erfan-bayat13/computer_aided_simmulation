# supermarket class
has arrival time, type, time for each sector
#### check the location tracking if its working

current location 
sections visited 
ready for check out 
has checked out
todo --> number of items to be added

## functions for client class
### generate client
generated random clients
todo --->  make it better

** returns ** SupermarketClient 

### arrival function (for now not being used because of load management)

#### todo check if this arrival and the other one in the supermarket class work the same

### decide next section
simple decisision making algo for seeing the next section to be selected 
todo -> check if the set in the innit is being added to properly 

### enter and exit section
Record entry/exit into a section.

both just record the metrics and change the flag for checked out

### get_total_shopping_time
sum of all total time

### journey summary
Get summary of client's shopping journey.

---> todo the journey summary for all the clients should be added in a csv file for after simmulation analysis

# supermarket class
init --> num servers for each section , queue capacity, rates, time, seed

active clients

todo --> make the service rates vary adn more relastic 

section are queue class *other is a infinte server queue*

##### FES *chose heap for my reasons* is initialised here and pushed the first heapq.heappush(self.FES, (0, "arrival", None))


## functions
### init metrics
#### mimin func called hour stats

### update hourly stats
todo --> see if you can make this be in a 24 hour window in every case 

Update hourly statistics with proper initialization
based on the event_type given the stats are added

### get availabe server
finds free server
### ger queue lenght
current queue lengths for all sections

### handle arrival
*this function is being used as the main arrivla function as is now*
this is inly for new customers
##### FES is being called here 
and used the genearte client fucntion to create new instances of customers
the we are using the lenght getters to assigne new section
##### FES is being handeld here as well

### section entry
handle section entry with proper server and queue management.
##### FES is being  called here

todo ---> handle the proper service time handeling 

### find_alternative_section
Find alternative section when original section is full
pass the client to other so they can be there until the queue is full

### handle section departure 
todo --> FES is not being handled in the other section

### check_customer_ready_for_checkout
flag handeling
### handle_customer_exit
handeling metrics and removing form the env
and updating the journey 

## run function
todo --->  check the time handleing
get time type data from FES
check type and pass to the right function

arrival
section entry
section departure then plotting the metrics
