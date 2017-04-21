package kr.ac.jejunu;

import javax.sql.DataSource;
import java.sql.*;

public class ProductDao {


    JdbcContext jdbcContext;

    public ProductDao() {

    }

    public void setJdbcContext(JdbcContext jdbcContext) {
        this.jdbcContext = jdbcContext;
    }



    public Product get(Long id) throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = new GetUserStatementStrategy(id);
        return jdbcContext.JdbcContextWithStatementForQuery(statementStrategy);
    }


    public void add(Product product) throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = new AddUserStatementStrtegy(product);
        jdbcContext.JdbcContextWithStatementForUpdate(statementStrategy);
    }



    public void update(Product product) throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = new UpdateUserStatementStrategy(product);
        jdbcContext.JdbcContextWithStatementForUpdate(statementStrategy);
    }

    public void delete(Long id)  throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = new DeleteUserStatementStrategy(id);
        jdbcContext.JdbcContextWithStatementForUpdate(statementStrategy);
    }
}
