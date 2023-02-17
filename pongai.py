import os
import neat
import pygame.display
from pong import Game
import pickle

class PongGame:
    def __init__(self,window,width,height):
        self.game = Game(window,width,height)
        self.left_padde = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball


    def train_ai(self,genome1,genome2,config):
        #Lines 17 and 18 create our networks
        net1 = neat.nn.FeedForwardNetwork.create(genome1,config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2,config)
        width,height = 700,500
        window = pygame.display.set_mode((width,height))
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            #Lines 28 and 31 pass in the features to our network and gets the output
            output1 = net1.activate((self.left_padde.y, self.ball.y, abs(self.left_padde.x - self.ball.x)))
            decision1 = output1.index(max(output1))
            #Lines 29 and 32 map the outputs of our networks to a index value such as 0,1 or 2....
            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))


            #Lines 36-47 handle what our networks should do with the outputs given
            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left = True, up = True)
            else:
                self.game.move_paddle(left = True,up = False)
            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left = False,up = True)
            else:
                self.game.move_paddle(left = False,up = False)
            game_info = self.game.loop()


            # self.game.draw(False,True)  42 & 43 Commented out for performance purposes
            # pygame.display.update()
            # Line 45 checks to see if each of the genomes playing each other have scored in this case it will break out of our loop
            if self.game.left_score >= 1 or self.game.right_score >= 1 or self.game.left_hits >=  50:
                self.calculate_fitness(genome1,genome2,game_info)
                break


    #Function to determine how well each genome performs
    def calculate_fitness(self,genome1,genome2,game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

    def test_ai(self,genome,config):
        width,height = 700,500
        window = pygame.display.set_mode((width,height))
        clock = pygame.time.Clock()
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        run = True
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left = True, up = True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left = True, up = False)

            output = net.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x-self.ball.x)))
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left = False, up = True)
            else:
                self.game.move_paddle(left = False, up = False)
            game_info = self.game.loop()
            self.game.draw(True,False)
            pygame.display.update()
        pygame.quit()


def eval_genomes(genomes,config):
    width,height = 700,500
    window = pygame.display.set_mode((width,height))

    #The block of code from 103-110 ensures that each genome plays each other once
    for i,(genome_id1,genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2,genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window,width,height)
            game.train_ai(genome1,genome2,config)


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint()
    p = neat.Population(config)
    p.add_reporter(neat.Checkpointer(1))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes,11)
    with open("best.pickle","wb") as f:
        pickle.dump(winner,f)


def test_ai(config):
    width,height = 700,500
    window = pygame.display.set_mode((width,height))
    with open("best.pickle","rb") as f:
        winner = pickle.load(f)
    game = PongGame(window,width,height)
    game.test_ai(winner,config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    test_ai(config)


