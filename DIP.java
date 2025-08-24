interface Keyboard {
    void type();
}

class WiredKeyboard implements Keyboard {
    public void type() {
        System.out.println("Typing with Wired Keyboard...");
    }
}

class WirelessKeyboard implements Keyboard {
    public void type() {
        System.out.println("Typing with Wireless Keyboard...");
    }
}

class Computer {
    private Keyboard keyboard;

    // Computer depends on abstraction (Keyboard)
    public Computer(Keyboard keyboard) {
        this.keyboard = keyboard;
    }

    public void start() {
        keyboard.type();
    }
}
