package kr.ac.jejunu;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

/**
 * Created by masinogns on 2017. 4. 21..
 */
public interface StatementStrategy {
    public PreparedStatement makeStatement(Connection connection) throws SQLException ;
}
