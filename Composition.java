// Engine class
class Engine {
    private String type;

    public Engine(String type) {
        this.type = type;
    }

    public void start() {
        System.out.println(type + " engine starting...");
    }
}

// Car class uses composition: Car has an Engine
class Car {
    private Engine engine;  // Car HAS an Engine
    private String model;

    public Car(String model, Engine engine) {
        this.model = model;
        this.engine = engine;
    }

    public void startCar() {
        System.out.println(model + " is ready.");
        engine.start();  // delegate work to Engine
    }
}

// Main
public class CompositionExample {
    public static void main(String[] args) {
        Engine engine = new Engine("V8");
        Car car = new Car("Mustang", engine);
        car.startCar();
    }
}
