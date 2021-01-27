from dqn import Trainer

trainer = Trainer()
model = trainer.train()

model.save_weights('./slither_model')





