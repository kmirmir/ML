package kr.ac.jejunu;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

/**
 * Created by masinogns on 2017. 4. 21..
 */
public class JejuConnectionMaker implements ConnectionMaker {
    String driverClass ;
    String url ;
    String username;
    String password;

    @Override
    public Connection getConnection() throws ClassNotFoundException, SQLException
    {

        Class.forName(driverClass);
        return DriverManager.getConnection(url, username, password);
    }

    public void setDriverClass(String driverClass) {
        this.driverClass = driverClass;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
