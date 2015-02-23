'''
Created on 18.2.2015

@author: tohekorh
'''
from threading import Thread
from threading import Semaphore
#from asyncio.tasks import sleep
import time 
import numpy as np

'''
ThreadMaster __init__ asetetaan maksimimaara kayettavia threads
windos rajoittaa maaran 255. linuksista en tieda.
Suosittelisin kayttamaan vahan yli koneen prosessorien maaran.

Ajettava funktio annetaan startFuction medotiin.
ja funktion parametrit args muuttujaan.

ThreadMaster tehdaan globali objecti.

Kun thredilla ajettava funktio loppuu on sen viimeisena tehtavana
kutsuttava ThreadMaster objectin done medotia.

Muista etta eri thredien kayttama alue ei saa muuttua toisen thredin
ajon seurauksena. Suojausta on kaytettava jos tallaisia tilanteita
ei voida valtaa.

Kun kaikki thredeilla ajettavat tehtavat on annettu  Threadmaster
objectille on kaynistajan jaatava odotamaan thredien valmistumista
waitUntilAllReady funktiion.

'''


class ThreadMaster:
    '''
    classdocs
    '''
    
    def __init__(self, maxThreads):
        self.maxThreads = maxThreads
        self.freeThreads = Semaphore(value=self.maxThreads)

    def waitUntilAllReady(self):

        for c in range(0, self.maxThreads ):
            self.freeThreads.acquire()

        for c in range(0, self.maxThreads ):
            self.freeThreads.release()


    def startFunction(self, fun, ar ):
        newThread = Thread(target=fun, args=ar)
        self.freeThreads.acquire()
        newThread.start()

    def done(self):
        self.freeThreads.release();

tm = ThreadMaster(2);

def testFun(forces, i, hilda = 'aa'):
    forces[i]   =   i
    j           =   0
    
    for i in range(int(1e6)):
        j = np.sqrt(i**2) 
    
    #time.sleep(10)
    
    print('testFun: ' + hilda + ' done')
    tm.done();

def Main():
    n   =   10
    forces  =    np.zeros(n)
    print('Main start')
    for i in range(n):
        tm.startFunction(fun=testFun, ar=(forces, i, 'paiva %i' %i,))
    #tm.startFunction(fun=testFun, ar=(forces, 1, 'paiva2',))
    #tm.startFunction(fun=testFun, ar=(forces, 2, 'paiva3',))
    tm.waitUntilAllReady()
    print forces
    print('Main end')

'''
if __name__ == '__main__':
    print('Maini alkaa')
    Main()
'''

print('Kojo kokj')
Main()
