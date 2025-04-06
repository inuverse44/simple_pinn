# entry point for the package
if __name__ == '__main__':
    import time
    from simple_PINN.simple_PINN import main_PINN
    start_time = time.time()
    main_PINN()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")