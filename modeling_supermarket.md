## supermarket model
## quesstion
The goal of the lab is to build a simulation model of the dynamics within a super-market.  The super-market is provided with a fresh-food desk (in which clients can buy personalized qualtities of cheese/ham/salami) and a  fresh-meat desk (butchery).   Clients  obtain service both at the  fresh-food desk and resh-meat desk, by interacting with a few assistants.

A few cash registers are running in parallel.

Try to build a model that achieves a reasonable trade-off between accuracy and complexity.  Which are the main dynamics that has to be modelled? Which are the details we can neglect?  How do we represent the process of picking items out of the shelves by customers? How to describe the queueing policy at  cash registers?

the idea is to have a supermarket as the entire system which has some queuing systems inside it.
the entities that are in this system are as follows
#### {butchery,freshfood,otherparts,cashregisters}
#### clients
from the client class we created earlier with a few adjustments 
client class has the following attributes
arrival_time, type={"fresh","butch","both","none"}, service_time = butch+fresh+other , remaining_time = service_time


## butchery and fresh food section

the approach is considering there two as seperate queues from my enviroment, they work on their own time with the number of assistants given to each.

each of them has a normal FCFS system.
only edge case to be considered will be explained in the run section below
(handleing what queue gets selected first)

## cash registery
this part again is a queuing system that has a FSFC approach.
only thing to consider is that maybe if the line is too full the client can go back to "other parts section" and basically waste time without crowding the queuign system

## run case
c1 enters with the following features: arrival_time, type, service_time, remaining_time
if type is fresh or butch or both{
    if any of the queus < pre defined treshold{
        go to that queue
        return remaining time
    }
    else{
        go to other parts

        reduce remaining time
        if butch{
            go to butch line
        }
        if fresh{
            goto fresh
        }
        if both{
            check the smallest line enter
        }
    }
}
else{
    go to other parts
}

here cash register queue is simulating 
FIFO queue

(edge cases for old ladies that dont think)
(can the arrival_rate and service_rate be dynamic and change with time? ----> arrival_rate a function of time, non homoginous possion process)
(some clients can return to other parts when the cash lines are too crowded)
(more workers at peak hours)