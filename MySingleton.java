
public final class MySingleton {
    private static final MySingleton INSTANCE = new MySingleton();

    private MySingleton() {}

    public static MySingleton getInstance() {
        return INSTANCE;
    }
}

/*
public final class MySingleton {
    private static volatile MySingleton instance = null;

    private MySingleton() {}
    
    public static MySingleton getInstance() {
        if (instance == null) {
            synchronized(MySingleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
*/