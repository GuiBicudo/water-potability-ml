from trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(
        data_path="data/water_potability.csv"
    )
