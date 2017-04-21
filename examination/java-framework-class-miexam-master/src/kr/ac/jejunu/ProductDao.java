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
        StatementStrategy statementStrategy = connection -> {
            PreparedStatement preparedStatement;
            preparedStatement = connection.prepareStatement("select * from product where id = ?");
            preparedStatement.setLong(1, id);
            return preparedStatement;
        };
        return jdbcContext.JdbcContextWithStatementForQuery(statementStrategy);
    }


    public void add(Product product) throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = connection -> {
            PreparedStatement preparedStatement;
            preparedStatement = connection.prepareStatement("insert into product (id, title, price) VALUES (?,?,?)");
            preparedStatement.setLong(1, product.getId());
            preparedStatement.setString(2, product.getTitle());
            preparedStatement.setInt(3, product.getPrice());
            preparedStatement.executeUpdate();
            return preparedStatement;
        };
        jdbcContext.JdbcContextWithStatementForUpdate(statementStrategy);
    }



    public void update(Product product) throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = connection -> {
            PreparedStatement preparedStatement;
            preparedStatement = connection.prepareStatement("update product set title = ?, price = ? where id = ?");
            preparedStatement.setString(1, product.getTitle());
            preparedStatement.setInt(2, product.getPrice());
            preparedStatement.setLong(3, product.getId());
            preparedStatement.executeUpdate();
            return preparedStatement;
        };
        jdbcContext.JdbcContextWithStatementForUpdate(statementStrategy);
    }

    public void delete(Long id)  throws ClassNotFoundException, SQLException {
        StatementStrategy statementStrategy = connection -> {
            PreparedStatement preparedStatement;
            preparedStatement = connection.prepareStatement("delete from product where id = ?");
            preparedStatement.setLong(1, id);
            preparedStatement.executeUpdate();
            return preparedStatement;
        };
        jdbcContext.JdbcContextWithStatementForUpdate(statementStrategy);
    }
}
